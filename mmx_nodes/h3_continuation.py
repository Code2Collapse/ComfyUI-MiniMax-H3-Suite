"""H3 Continuation Next / Commit - resumable, unbounded H3 generation on an on-disk session (A16 Part IV).

Original implementation, GPL-3.0. Design: docs/h3_continuation_design.md. Replaces the H3 Motion Context chain
(placeholder Chain node, manual clip_index, negative audio anchors, 8.3 ms audio offset per segment).

Next (before the sampler): the previous ACCEPTED segment's last frames go into the head of this segment's latent as
their exact tokens, pinned by the denoise mask (0 = keep). Commit (after the decoders): checks the head came back
unchanged, keeps only the new frames / sound (cut on global time), stores the new tail with a checksum, measures drift,
and accepts, holds for approval, or rejects the segment. Queue the graph again (or set a batch count) for the next one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_session as hs  # noqa: E402

DECISIONS = ("keep waiting", "accept the pending segment", "reject the pending segment")
APPROVALS = ("accept unless drift", "accept", "always ask")
HEAD_TOLERANCE = 1e-3


def _base_dir() -> str:
    import folder_paths
    return os.path.join(folder_paths.get_output_directory(), "h3_continuation")


def _session_digest(name: str) -> str:
    import hashlib
    try:
        path = os.path.join(_base_dir(), hs.safe_name(name), "session.json")
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except (OSError, hs.SessionError):
        return "new"


def _sessions_mtime() -> int:
    base = _base_dir()
    newest = 0
    if os.path.isdir(base):
        for d in os.listdir(base):
            f = os.path.join(base, d, "session.json")
            if os.path.isfile(f):
                newest = max(newest, os.stat(f).st_mtime_ns)
    return newest


def _streams(latent: dict, what: str):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError(f"H3 Continuation: {what} is not a MiniMax H3 picture + sound latent.")
    video, audio = samples.unbind()[:2]
    if video.ndim != 5 or video.shape[1] != 24 or audio.ndim != 4:
        raise ValueError(f"H3 Continuation: {what} has shapes {tuple(video.shape)} / {tuple(audio.shape)}; "
                         "expected video [1, 24, T, H/16, W/16] and audio [1, 32, 2, T40].")
    return video, audio


def _nested(parts):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(parts))


class MiniMaxH3_ContinuationNext(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContinuationNext",
            display_name="H3 Continuation Next",
            category="MiniMax H3/Long",
            description=(
                "Start the next segment of a long H3 video. The last frames of the previous ACCEPTED segment are put "
                "at the start of this segment as their exact latent tokens and held there by the noise mask, so the "
                "model continues the shot instead of restarting it. Wire the latent it outputs to the sampler and the "
                "segment output to H3 Continuation Commit. Queue the graph once per segment; the session on disk "
                "remembers where it is, survives a restart, and never changes frames it has already accepted."),
            inputs=[
                io.String.Input("session", default="h3_long",
                                tooltip="Name of the long video. Its files go to output/h3_continuation/<name>."),
                io.Conditioning.Input("positive", tooltip="The segment's conditioning (prompt, references, guides)."),
                io.Latent.Input("latent", tooltip="An empty H3 picture + sound latent of ONE segment (e.g. from Empty "
                                                  "MiniMax H3 AV Latent or Reference to Video): it sets the size and "
                                                  "the segment length."),
                io.Combo.Input("overlap_frames", options=[str(o) for o in hs.OVERLAPS], default="39",
                               tooltip="Frames of the previous segment the model continues from. 39 keeps sound and "
                                       "picture on the same boundary; 22 / 56 are allowed (the report gives the "
                                       "boundary offset). More = smoother, less new footage per segment."),
                io.Combo.Input("decision", options=list(DECISIONS), default=DECISIONS[0],
                               tooltip="For a segment that Commit held for approval: accept it (it joins the video) "
                                       "or reject it (its frames are generated again). 'keep waiting' stops here."),
                io.Boolean.Input("restart", default=False,
                                 tooltip="Archive this session (renamed with the date) and start again from frame 0."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="This segment's latent, with the pinned head."),
                io.Conditioning.Output(display_name="positive", tooltip="The conditioning, minus guides inside the "
                                                                        "pinned head (it is already fixed)."),
                io.String.Output(display_name="segment", tooltip="For H3 Continuation Commit."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, session="h3_long", overlap_frames="39", decision=DECISIONS[0], restart=False,
                           **_):
        # the state lives in session.json: it changes after every commit, so the next queue starts the next segment;
        # an unchanged session (e.g. a segment still awaiting a decision) is served from the cache
        return f"{_session_digest(session)}|{overlap_frames}|{decision}|{restart}"

    @classmethod
    def execute(cls, session, positive, latent, overlap_frames="39", decision=DECISIONS[0],
                restart=False) -> io.NodeOutput:
        video, audio = _streams(latent, "latent")
        T, h, w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
        L = hs.frames_for_rows(T)
        overlap = int(overlap_frames)
        s = hs.Session.open(_base_dir(), session, restart=bool(restart))
        plan = s.plan_next(L, overlap, {"latent_hw": [h, w], "segment_frames": L}, f"{T}x{h}x{w}", decision)
        seg = plan.segment
        notes = []
        out_latent = {k: v for k, v in latent.items() if k != "noise_mask"}
        pos_out = positive
        if plan.tail is None:
            seg["audio_origin_ticks"] = 0
            notes.append(f"segment {seg['id']}: the first, frames 0-{seg['end'] - 1}, nothing pinned")
        else:
            prev = s.last_accepted()
            slack = hs.audio_slack_seconds(L, int(audio.shape[-1]))
            seg["audio_slack_s"] = slack
            v_rows, a_ticks, t0 = hs.tail_slice(prev, plan.tail, overlap, seg["start"], int(prev["audio_ticks"]),
                                                slack)
            R = int(v_rows.shape[2])
            A = min(int(a_ticks.shape[-1]), int(audio.shape[-1]))
            new_v = video.clone()
            new_v[:, :, :R] = v_rows.to(new_v)
            new_a = audio.clone()
            new_a[..., :A] = a_ticks[..., :A].to(new_a)
            vm = torch.ones((1, 1, T, h, w))
            vm[:, :, :R] = 0.0
            am = torch.ones((1, 1, 2, int(audio.shape[-1])))
            am[..., :A] = 0.0
            out_latent["samples"] = _nested((new_v, new_a))
            out_latent["noise_mask"] = _nested((vm, am))
            seg["audio_origin_ticks"] = int(prev.get("audio_origin_ticks", 0)) + t0
            seg["pinned_rows"], seg["pinned_ticks"] = R, A
            err_ms = (hs.audio_origin_seconds(seg) - seg["start"] / hs.FPS) * 1000.0
            seg["audio_offset_ms"] = round(err_ms, 3)
            pos_out, dropped = _drop_head_guides(positive, overlap)
            notes.append(f"segment {seg['id']}: frames {seg['start']}-{seg['end'] - 1}, the first {overlap} pinned "
                         f"from segment {prev['id']} ({R} latent rows, {A} audio ticks); audio starts "
                         f"{err_ms:+.1f} ms from the picture (kept within {hs.phase_bound_ms(slack):.1f} ms, "
                         "never accumulating)")
            if dropped:
                notes.append(f"dropped {dropped} guide(s) inside the pinned head - those frames are already fixed")
        if seg.get("retry_of") is not None:
            notes.append(f"this generates the interval of segment {seg['retry_of']} again")
        out_latent["mmx_continuation"] = {"session": s.state["name"], "segment": seg["id"]}
        s.save()
        info = {"session": s.state["name"], "root": s.root, "id": seg["id"], "start": seg["start"],
                "end": seg["end"], "overlap": seg["overlap"]}
        report = "\n".join(notes + [f"accepted so far: {s.state['accepted_frames']} frames "
                                    f"({s.state['accepted_frames'] / hs.FPS:.2f} s)"])
        return io.NodeOutput(out_latent, pos_out, json.dumps(info), report, ui={"text": [report]})


def _drop_head_guides(positive, overlap: int):
    out, dropped = [], 0
    for emb, extra in positive:
        d = dict(extra)
        kfs = d.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if int(kf.get("resolved_frame_index", 0)) >= overlap]
            dropped += len(kfs) - len(kept)
            d["minimax_keyframes"] = kept
        out.append([emb, d])
    return out, dropped


class MiniMaxH3_ContinuationCommit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ContinuationCommit",
            display_name="H3 Continuation Commit",
            category="MiniMax H3/Long",
            description=(
                "Finish a segment started by H3 Continuation Next: check the pinned head came back unchanged, keep "
                "only the new frames and sound (cut on the video's global time, so sound never drifts), store the "
                "new tail, measure drift against the first segment, and accept the segment, hold it for your "
                "approval, or reject it. Accepted frames go to output/h3_continuation/<name>/frames as a numbered "
                "16-bit PNG sequence, the sound to audio/ as one WAV per segment."),
            inputs=[
                io.String.Input("segment", force_input=True, tooltip="From H3 Continuation Next."),
                io.Latent.Input("samples", tooltip="The sampler's output latent for this segment."),
                io.Image.Input("images", tooltip="The decoded frames of the WHOLE segment (VAE Decode of samples)."),
                io.Audio.Input("audio", optional=True, tooltip="The decoded sound of the whole segment."),
                io.Combo.Input("approval", options=list(APPROVALS), default=APPROVALS[0],
                               tooltip="accept unless drift: hold the segment when a drift limit is crossed. accept: "
                                       "always. always ask: hold every segment until H3 Continuation Next's decision."),
                io.Float.Input("max_colour_drift", default=6.0, min=0.5, max=50.0, step=0.5,
                               tooltip="Largest colour change (CIE Lab dE of the mean) from segment 0."),
                io.Float.Input("min_sharpness", default=0.6, min=0.05, max=1.0, step=0.05,
                               tooltip="Smallest sharpness (Laplacian variance) as a fraction of segment 0."),
                io.Float.Input("max_seam_jump", default=3.0, min=1.0, max=50.0, step=0.5,
                               tooltip="Largest change across the seam, in multiples of the clip's usual frame change."),
            ],
            outputs=[
                io.Image.Output(display_name="new_frames", tooltip="The frames this segment added."),
                io.Audio.Output(display_name="new_audio", tooltip="The sound this segment added."),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="state_json", tooltip="The session state after this segment."),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, approval=APPROVALS[0], max_colour_drift=6.0, min_sharpness=0.6, max_seam_jump=3.0,
                           **_):
        # 'segment' is linked (not visible here): key on the newest session file, which Next rewrites every segment
        return f"{_sessions_mtime()}|{approval}|{max_colour_drift}|{min_sharpness}|{max_seam_jump}"

    @classmethod
    def execute(cls, segment, samples, images, audio=None, approval=APPROVALS[0], max_colour_drift=6.0,
                min_sharpness=0.6, max_seam_jump=3.0) -> io.NodeOutput:
        try:
            info = json.loads(segment)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("H3 Continuation Commit: 'segment' must come from H3 Continuation Next.") from exc
        s = hs.Session.open(os.path.dirname(info["root"]), info["session"])
        seg = s.get(int(info["id"]))
        if seg["status"] != "running":
            raise ValueError(f"H3 Continuation Commit: segment {seg['id']} is '{seg['status']}', not running - "
                             "queue the graph again to start a new segment.")
        try:
            report, new_frames, new_audio = _commit(s, seg, samples, images, audio, approval,
                                                    max_colour_drift, min_sharpness, max_seam_jump)
        except Exception as exc:
            seg["status"] = "failed"
            seg["note"] = f"commit failed: {exc}"
            s.save()
            raise
        state = {k: s.state[k] for k in ("name", "accepted_frames", "audio_samples", "sample_rate")}
        state["segments"] = [{k: g.get(k) for k in ("id", "start", "end", "status", "out_frames", "note")}
                             for g in s.segments]
        return io.NodeOutput(new_frames, new_audio, report, json.dumps(state), ui={"text": [report]})


def _commit(s: hs.Session, seg: dict, samples, images, audio, approval, colour_max, sharp_min, seam_max):
    video, a_lat = _streams(samples, "samples")
    L = seg["end"] - seg["start"]
    if images.shape[0] != L:
        raise ValueError(f"images has {images.shape[0]} frames; this segment is {L} frames ({seg['start']}-"
                         f"{seg['end'] - 1}). Decode the whole segment's latent.")
    notes = []
    # 1. the pinned head came back unchanged (transport fidelity)
    if seg["parent"] is not None:
        prev = s.get(seg["parent"])
        tail = s.load_tail(prev)
        v_rows, a_ticks, _ = hs.tail_slice(prev, tail, seg["overlap"], seg["start"], int(prev["audio_ticks"]),
                                           float(seg.get("audio_slack_s", 0.0)))
        R, A = seg["pinned_rows"], seg["pinned_ticks"]
        dv = float((video[:, :, :R].float().cpu() - v_rows.float()).abs().max())
        da = float((a_lat[..., :A].float().cpu() - a_ticks[..., :A].float()).abs().max())
        seg["head_max_diff"] = [dv, da]
        if max(dv, da) > HEAD_TOLERANCE:
            raise ValueError(f"the pinned head changed during sampling (largest change {max(dv, da):.4g}): the "
                             "sampler did not honour the noise mask. Feed H3 Continuation Next's latent straight to "
                             "the sampler (no latent node in between that drops 'noise_mask').")
        notes.append(f"pinned head unchanged (largest latent change {max(dv, da):.2g})")
    # 2. store the new tail (exact tokens, checksum)
    T = int(video.shape[2])
    rows = min(T, hs.TAIL_ROWS)
    if (T - rows) % 5:
        rows = T
    s.store_tail(seg, video[:, :, T - rows:], a_lat[..., max(0, int(a_lat.shape[-1]) - hs.TAIL_TICKS):])
    seg["audio_ticks"] = int(a_lat.shape[-1])
    # 3. the new frames, on global numbering
    a, b = hs.new_frame_range(seg, s.state["accepted_frames"])
    new = images[a - seg["start"]: b - seg["start"]]
    seg["out_frames"] = [a, b]
    prev_last = None
    if a > 0:
        p = os.path.join(s.root, "frames", f"frame_{a:06d}.png")
        if os.path.isfile(p):
            prev_last = hs.read_png16(p)
    metrics = hs.drift_metrics(new.float().cpu(), s.state.get("baseline"), prev_last)
    seg["metrics"] = metrics
    # 4. sound, cut on global time
    new_audio = None
    if audio is not None:
        sr = int(audio["sample_rate"])
        if s.state.get("sample_rate") not in (None, sr):
            raise ValueError(f"the sound is {sr} Hz but the session's is {s.state['sample_rate']} Hz.")
        wav = audio["waveform"][0] if audio["waveform"].ndim == 3 else audio["waveform"]
        origin = int(round(hs.audio_origin_seconds(seg) * sr))
        lo, hi = hs.samples_at(a, sr) - origin, hs.samples_at(b, sr) - origin
        piece = wav[..., max(0, lo):max(0, hi)]
        want = hs.samples_at(b, sr) - hs.samples_at(a, sr)
        if lo < 0:
            piece = torch.cat([torch.zeros(piece.shape[0], -lo, dtype=piece.dtype), piece], dim=-1)
        if piece.shape[-1] < want:
            piece = torch.cat([piece, torch.zeros(piece.shape[0], want - piece.shape[-1], dtype=piece.dtype)], -1)
        piece = piece[..., :want].cpu()
        new_audio = {"waveform": piece[None], "sample_rate": sr}
        s.state["sample_rate"] = sr
        seg["audio_samples_end"] = hs.samples_at(b, sr)
    # 5. decide
    problems = hs.drift_problems(metrics, colour_max, sharp_min, seam_max)
    if approval == "always ask":
        status = "awaiting approval"
        seg["note"] = "held for your decision (approval = always ask)"
    elif approval == "accept unless drift" and problems:
        status = "awaiting approval"
        seg["note"] = "; ".join(problems)
    else:
        status = "accepted"
    pend = os.path.join(s.root, "pending", f"seg_{seg['id']:04d}")
    os.makedirs(pend, exist_ok=True)
    for i in range(new.shape[0]):
        hs.write_png16(os.path.join(pend, f"{i:06d}.png"), new[i])
    if new_audio is not None:
        hs.write_wav(os.path.join(pend, "audio.wav"), new_audio["waveform"][0], new_audio["sample_rate"])
    if status == "accepted":
        s._place(seg, pend)
        import shutil
        shutil.rmtree(pend, ignore_errors=True)
        if s.state.get("baseline") is None:
            s.state["baseline"] = metrics["stats"]
    seg["status"] = status
    s.save()
    m = metrics
    notes.append(f"segment {seg['id']}: +{b - a} frames ({a}-{b - 1}) -> {status}"
                 + (f" ({seg['note']})" if status != "accepted" else ""))
    if m.get("colour_delta_e") is not None:
        notes.append(f"drift vs segment 0: colour {m['colour_delta_e']:.2f} dE, sharpness x{m['sharpness_ratio']:.2f}")
    if m.get("seam_jump_ratio") is not None:
        notes.append(f"seam: {m['seam_jump_ratio']:.2f}x the usual frame change; frozen frames {m['frozen_frames']}")
    notes.append(f"accepted total: {s.state['accepted_frames']} frames ({s.state['accepted_frames'] / hs.FPS:.2f} s) "
                 f"in {s.root}")
    if new_audio is None:
        new_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 32000}
    return "\n".join(notes), new.cpu(), new_audio


__all__ = ["MiniMaxH3_ContinuationNext", "MiniMaxH3_ContinuationCommit"]
