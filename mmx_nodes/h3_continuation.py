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
import math
import os
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_session as hs  # noqa: E402
from mmx_utils import h3_video_slice as vs  # noqa: E402

DECISIONS = ("keep waiting", "accept the pending segment", "reject the pending segment")
APPROVALS = ("accept unless drift", "accept", "always ask")
SOUNDS = ("generate", "keep the source video's sound", "keep the soundtrack", "regenerate where masked")
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
                "remembers where it is, survives a restart, and never changes frames it has already accepted. "
                "Long inputs (a plate to edit, its mask video, a control video, a soundtrack) are cut to each "
                "segment here: only the segment's frames are decoded."),
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
                io.Vae.Input("video_vae", optional=True,
                             tooltip="The H3 video VAE - needed with source_video (the plate is encoded per segment)."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="The H3 audio VAE - needed when sound is kept (source sound or soundtrack)."),
                io.Video.Input("source_video", optional=True,
                               tooltip="A long plate to EDIT (Load Video). Each segment takes its own frames from it; "
                                       "where source_mask is black the plate is kept. Must have the latent's size."),
                io.Video.Input("source_mask", optional=True,
                               tooltip="The plate's mask as a video (white = regenerate), same length and size as "
                                       "source_video."),
                io.Video.Input("control_video", optional=True,
                               tooltip="A long control video (pose / depth / edges render of the whole film). Each "
                                       "segment's frames come out of segment_control for H3 Mask-Aware ControlNet."),
                io.Audio.Input("soundtrack", optional=True,
                               tooltip="A long soundtrack the video follows (sound = keep the soundtrack). The session "
                                       "ends with the soundtrack."),
                io.Combo.Input("sound", options=list(SOUNDS), default=SOUNDS[0], optional=True,
                               tooltip="generate: the model makes the sound. keep the source video's sound / keep the "
                                       "soundtrack: that sound is held fixed and the picture follows it (lip sync, "
                                       "music). regenerate where masked: the source sound is kept except where the "
                                       "mask covers a frame."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="This segment's latent, with the pinned head."),
                io.Conditioning.Output(display_name="positive", tooltip="The conditioning, minus guides inside the "
                                                                        "pinned head (it is already fixed)."),
                io.String.Output(display_name="segment", tooltip="For H3 Continuation Commit."),
                io.String.Output(display_name="report"),
                io.Image.Output(display_name="segment_source", tooltip="This segment's frames of source_video (for "
                                                                       "an exact composite before Commit). Empty "
                                                                       "unless source_video is connected."),
                io.Mask.Output(display_name="segment_mask", tooltip="This segment's frames of source_mask (all white "
                                                                  "with source_video and no mask). Empty without "
                                                                  "source_video."),
                io.Image.Output(display_name="segment_control", tooltip="This segment's frames of control_video. Empty "
                                                                     "unless control_video is connected."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, session="h3_long", overlap_frames="39", decision=DECISIONS[0], restart=False,
                           sound=SOUNDS[0], **_):
        # the state lives in session.json: it changes after every commit, so the next queue starts the next segment;
        # an unchanged session (e.g. a segment still awaiting a decision) is served from the cache
        return f"{_session_digest(session)}|{overlap_frames}|{decision}|{restart}|{sound}"

    @classmethod
    def execute(cls, session, positive, latent, overlap_frames="39", decision=DECISIONS[0], restart=False,
                video_vae=None, audio_vae=None, source_video=None, source_mask=None, control_video=None,
                soundtrack=None, sound=SOUNDS[0]) -> io.NodeOutput:
        video, audio = _streams(latent, "latent")
        T, h, w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
        L = hs.frames_for_rows(T)
        overlap = int(overlap_frames)
        _check_sources(sound, video_vae, audio_vae, source_video, source_mask, soundtrack)
        s = hs.Session.open(_base_dir(), session, restart=bool(restart))
        if s.state.get("complete"):
            raise hs.SessionError(f"The session '{s.state['name']}' is complete: all {s.state['accepted_frames']} "
                                  f"frames are in {os.path.join(s.root, 'frames')}. Set restart to start again.")
        plan = s.plan_next(L, overlap, {"latent_hw": [h, w], "segment_frames": L}, f"{T}x{h}x{w}", decision)
        seg = plan.segment
        try:
            built = _build_segment(s, plan, video, audio, L, overlap, video_vae, audio_vae, source_video,
                                   source_mask, control_video, soundtrack, sound)
        except _Complete as done:
            s.segments.remove(seg)
            s.state["complete"] = True
            s.save()
            raise hs.SessionError(str(done)) from None
        except Exception as exc:
            seg["status"] = "failed"
            seg["note"] = f"could not be prepared: {exc}"
            s.save()
            raise
        out_latent = {k: v for k, v in latent.items() if k not in ("noise_mask", "mmx_geometry")}
        out_latent["samples"] = _nested((built["video"], built["audio"]))
        if built["mask"] is not None:
            out_latent["noise_mask"] = _nested(built["mask"])
        notes = built["notes"]
        pos_out = positive
        if plan.tail is not None:
            pos_out, dropped = _drop_head_guides(positive, overlap)
            if dropped:
                notes.append(f"dropped {dropped} guide(s) inside the pinned head - those frames are already fixed")
        if seg.get("retry_of") is not None:
            notes.append(f"this generates the interval of segment {seg['retry_of']} again")
        out_latent["mmx_continuation"] = {"session": s.state["name"], "segment": seg["id"]}
        s.save()
        info = {"session": s.state["name"], "root": s.root, "id": seg["id"], "start": seg["start"],
                "end": seg["end"], "overlap": seg["overlap"], "source_end": seg.get("source_end")}
        report = "\n".join(notes + [f"accepted so far: {s.state['accepted_frames']} frames "
                                    f"({s.state['accepted_frames'] / hs.FPS:.2f} s)"])
        return io.NodeOutput(out_latent, pos_out, json.dumps(info), report, built["source"], built["pixel_mask"],
                             built["control"], ui={"text": [report]})


class _Complete(Exception):
    """The source has no frames after the accepted ones: the long video is finished."""


def _check_sources(sound, video_vae, audio_vae, source_video, source_mask, soundtrack):
    """Plain refusals before the session is touched."""
    if sound not in SOUNDS:
        raise ValueError(f"H3 Continuation Next: unknown sound choice {sound!r}.")
    if source_video is not None and video_vae is None:
        raise ValueError("H3 Continuation Next: source_video is connected - connect the H3 video VAE to video_vae "
                         "(each segment's plate frames are encoded here).")
    if source_mask is not None and source_video is None:
        raise ValueError("H3 Continuation Next: source_mask needs source_video (the mask says which plate pixels "
                         "are regenerated).")
    if sound in (SOUNDS[1], SOUNDS[3]) and source_video is None:
        raise ValueError(f"H3 Continuation Next: sound = '{sound}' needs source_video.")
    if sound == SOUNDS[3] and source_mask is None:
        raise ValueError("H3 Continuation Next: sound = 'regenerate where masked' needs source_mask.")
    if sound == SOUNDS[2] and soundtrack is None:
        raise ValueError("H3 Continuation Next: sound = 'keep the soundtrack' needs the soundtrack input.")
    if sound != SOUNDS[0] and audio_vae is None:
        raise ValueError(f"H3 Continuation Next: sound = '{sound}' needs the H3 audio VAE on audio_vae.")


def _build_segment(s, plan, video, audio, L, overlap, video_vae, audio_vae, source_video, source_mask,
                   control_video, soundtrack, sound) -> dict:
    """The segment's latent (+ noise mask) and its slices of the long inputs."""
    from mmx_nodes.source_av_latent import _encode_audio, build_source_av_latent, fit_audio_ticks
    seg = plan.segment
    start = seg["start"]
    T, h, w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    TA = int(audio.shape[-1])
    accepted = s.state["accepted_frames"] if plan.tail is not None else 0
    notes = []
    # the audio origin on global time (tick grid of the previous segment)
    slack = hs.audio_slack_seconds(L, TA)
    v_rows = a_ticks = None
    if plan.tail is None:
        seg["audio_origin_ticks"] = 0
        notes.append(f"segment {seg['id']}: the first, frames 0-{seg['end'] - 1}, nothing pinned")
    else:
        prev = s.last_accepted()
        seg["audio_slack_s"] = slack
        v_rows, a_ticks, t0 = hs.tail_slice(prev, plan.tail, overlap, start, int(prev["audio_ticks"]), slack)
        seg["audio_origin_ticks"] = int(prev.get("audio_origin_ticks", 0)) + t0
        err_ms = (hs.audio_origin_seconds(seg) - start / hs.FPS) * 1000.0
        seg["audio_offset_ms"] = round(err_ms, 3)
        notes.append(f"segment {seg['id']}: frames {start}-{seg['end'] - 1}, the first {overlap} pinned from segment "
                     f"{prev['id']} ({int(v_rows.shape[2])} latent rows, {min(int(a_ticks.shape[-1]), TA)} audio "
                     f"ticks); audio starts {err_ms:+.1f} ms from the picture (kept within "
                     f"{hs.phase_bound_ms(slack):.1f} ms, never accumulating)")
    origin_s = hs.audio_origin_seconds(seg)
    base_v, base_a = video, audio
    vmask = amask = None
    source = pixel_mask = control = None
    end = None
    if source_video is not None:
        rate, note = vs.check_rate(source_video, "source_video")
        if note:
            notes.append(note)
        keep_src_sound = sound in (SOUNDS[1], SOUNDS[3])
        if keep_src_sound and rate != 24:
            raise ValueError("H3 Continuation Next: the source video is not exactly 24 fps, so its sound cannot be "
                             "kept in sync - use sound = generate, or a 24 fps plate.")
        frames, got, win_audio = vs.read_frames(source_video, start, L, with_audio=keep_src_sound)
        if got == 0 or start + got <= accepted:
            raise _Complete(f"The source video ends at frame {start + got}: every frame is done "
                            f"({s.state['accepted_frames']} accepted). Set restart to start again.")
        if got < L:
            end = start + got
            notes.append(f"the source video ends at frame {end}: the last {L - got} frames of this segment are "
                         "padding and are not kept")
        if source_mask is not None:
            mframes, mgot, _ = vs.read_frames(source_mask, start, L)
            if mgot < got:
                raise ValueError(f"H3 Continuation Next: source_mask has only {start + mgot} frames but the plate "
                                 f"has at least {start + got}. Give a mask frame for every plate frame.")
            pixel_mask = vs.mask_from_frames(mframes)
            if tuple(pixel_mask.shape[1:]) != tuple(frames.shape[1:3]):
                raise ValueError(f"H3 Continuation Next: source_mask is {pixel_mask.shape[2]}x{pixel_mask.shape[1]} "
                                 f"but source_video is {frames.shape[2]}x{frames.shape[1]}.")
            if got < L:
                pixel_mask[got:] = 0.0
        seg_audio = None
        if keep_src_sound:
            if win_audio is None:
                raise ValueError("H3 Continuation Next: the source video has no sound to keep - use sound = "
                                 "generate or keep the soundtrack.")
            seg_audio = vs.shift_audio(win_audio, origin_s - start / hs.FPS, TA / hs.AUDIO_TICKS_PER_SECOND)
        elif sound == SOUNDS[2]:
            seg_audio = vs.cut_audio(soundtrack, origin_s, TA / hs.AUDIO_TICKS_PER_SECOND)
        mode = {SOUNDS[0]: "regenerate", SOUNDS[1]: "keep", SOUNDS[2]: "keep", SOUNDS[3]: "follow mask"}[sound]
        lat, rep = build_source_av_latent(frames, video_vae, audio_vae if seg_audio is not None else None, seg_audio,
                                          pixel_mask, mode, "refuse")
        sv, sa = lat["samples"].unbind()[:2]
        if tuple(sv.shape[2:]) != (T, h, w):
            raise ValueError(f"H3 Continuation Next: the source video encodes to {tuple(sv.shape[2:])} latent "
                             f"(frames x height x width / 16) but the segment latent is {(T, h, w)}. Set the empty "
                             f"latent's width / height to the plate's ({frames.shape[2]}x{frames.shape[1]}).")
        smv, sma = lat["noise_mask"].unbind()[:2]
        sa, _ = fit_audio_ticks(sa, TA)
        sma = torch.nn.functional.pad(sma, (0, max(0, TA - int(sma.shape[-1]))), value=1.0)[..., :TA]
        base_v, base_a = sv.to(video), sa.to(audio)
        vmask, amask = smv.float().cpu(), sma.float().cpu()
        source = frames
        notes.append("source: " + rep.replace("\n", "; "))
    elif sound == SOUNDS[2]:
        enc = _encode_audio(audio_vae, vs.cut_audio(soundtrack, origin_s, TA / hs.AUDIO_TICKS_PER_SECOND))
        enc, _ = fit_audio_ticks(enc, TA)
        base_a = enc.to(audio)
        vmask = torch.ones((1, 1, T, h, w))
        amask = torch.zeros((1, 1, 2, TA))
    if sound == SOUNDS[2]:
        sound_end = int(math.ceil(vs.audio_seconds(soundtrack) * hs.FPS - 1e-9))
        if sound_end <= accepted:
            raise _Complete(f"The soundtrack ends at frame {sound_end}: every frame is done "
                            f"({s.state['accepted_frames']} accepted). Set restart to start again.")
        if sound_end < seg["end"]:
            end = sound_end if end is None else min(end, sound_end)
            notes.append(f"the soundtrack ends at frame {sound_end}: frames after it are not kept")
        notes.append("sound: the soundtrack is held fixed; the picture follows it")
    if end is not None:
        seg["source_end"] = int(end)
    if control_video is not None:
        _, cnote = vs.check_rate(control_video, "control_video")
        control, cgot, _ = vs.read_frames(control_video, start, L)
        if cgot < L and (end is None or start + cgot < end):
            notes.append(f"control_video ends at frame {start + cgot}: its last frame is held for the rest")
        if control.shape[1:3] != (h * 16, w * 16):
            notes.append(f"control_video is {control.shape[2]}x{control.shape[1]}, the segment "
                         f"{w * 16}x{h * 16}: the ControlNet node fits it (cover crop, or crop_bboxes)")
    # the pinned head
    new_v, new_a = base_v, base_a
    if plan.tail is not None:
        R = int(v_rows.shape[2])
        A = min(int(a_ticks.shape[-1]), TA)
        new_v = base_v.clone()
        new_v[:, :, :R] = v_rows.to(new_v)
        new_a = base_a.clone()
        new_a[..., :A] = a_ticks[..., :A].to(new_a)
        vmask = torch.ones((1, 1, T, h, w)) if vmask is None else vmask.clone()
        amask = torch.ones((1, 1, 2, TA)) if amask is None else amask.clone()
        vmask[:, :, :R] = 0.0
        amask[..., :A] = 0.0
        seg["pinned_rows"], seg["pinned_ticks"] = R, A
    mask = None if vmask is None else (vmask, amask)
    if source is not None and pixel_mask is None:
        pixel_mask = torch.ones(source.shape[:3])          # no mask video: the whole plate is regenerated
    # unconnected long inputs give None (never stand-in frames): a node fed from them sees "not connected"
    return {"video": new_v, "audio": new_a, "mask": mask, "notes": notes, "source": source,
            "pixel_mask": pixel_mask, "control": control}


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
    if seg.get("source_end") is not None:             # the plate / soundtrack ends inside this segment
        b = min(b, int(seg["source_end"]))
        if b < seg["end"]:
            notes.append(f"the source ends at frame {b}: the last {seg['end'] - b} frames (padding) are not kept")
    new = images[a - seg["start"]: b - seg["start"]]
    seg["out_frames"] = [a, b]
    prev_last = None
    if a > 0:
        p = os.path.join(s.root, "frames", f"frame_{a:06d}.png")
        if os.path.isfile(p):
            prev_last = hs.read_png16(p)
    metrics = hs.drift_metrics(new.float().cpu(), s.state.get("baseline"), prev_last)
    # the overlap (global frames start..a-1) was decoded again here: compare it with the frames already accepted
    # (the sequence is numbered from 1: global frame i is frame_{i+1})
    head_paths = [os.path.join(s.root, "frames", f"frame_{i + 1:06d}.png") for i in range(seg["start"], a)]
    if head_paths and all(os.path.isfile(p) for p in head_paths):
        accepted = torch.stack([hs.read_png16(p) for p in head_paths])
        metrics.update(hs.overlap_agreement(images[: a - seg["start"]].float().cpu(), accepted))
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
    if m.get("overlap_psnr_db") is not None:
        notes.append(f"overlap re-decoded here vs the accepted frames: {m['overlap_psnr_db']:.1f} dB over "
                     f"{m['overlap_frames']} frames (worst frame {m['overlap_worst_frame_psnr_db']:.1f} dB)")
    summary = hs.continuity_summary(s.segments)
    if summary:
        notes.append(summary)
    notes.append(f"accepted total: {s.state['accepted_frames']} frames ({s.state['accepted_frames'] / hs.FPS:.2f} s) "
                 f"in {s.root}")
    if new_audio is None:
        new_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 32000}
    return "\n".join(notes), new.cpu(), new_audio


__all__ = ["MiniMaxH3_ContinuationNext", "MiniMaxH3_ContinuationCommit"]
