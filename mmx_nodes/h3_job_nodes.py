"""H3 job nodes: Task Spec, Compatibility Validator, Segment Planner, Job Controller, QC Gate, Job Commit (owner A21
Revision 3 Parts 7-8; ledger L9.75). On the engine in mmx_utils/h3_job.py.

One queue click runs a whole job: the Controller hands this run the next segment, the graph generates it, the QC
Gate judges it, and the Commit records it and - when more is left and nothing needs a person - posts the SAME prompt
to the server's own POST /prompt, so the next run takes the next segment (Rev 3 8.5 "users must not manually queue
every continuation segment"). Guards: the job's state (paused / review / completed stop it), a chain cap
(``max_auto_runs``), the kill switch ``H3_AUTO_CONTINUE=0``.

Accepted frames are kept as 16-bit PNGs per segment attempt (exact, any length), numbered by GLOBAL frame; the final
movie is assembled from the chosen takes in plan order, streamed (one chunk in memory).

Original nodes, GPL-3.0. Design: docs/evidence/A21/phase2/DESIGN_L9.75_job_engine.md in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_constants as HC  # noqa: E402
from mmx_utils import h3_job as J  # noqa: E402
from mmx_vendor.c2c_core.timeline import ledger as L  # noqa: E402
from mmx_vendor.c2c_core.video import media as _media  # noqa: E402
from mmx_vendor.c2c_core.video.writer import StreamWriter  # noqa: E402

from .media_window import C2C_VIDEO, is_handle  # noqa: E402
from .segment_io import job_folder  # noqa: E402

TASKS = ("edit (source-driven)", "generate (source-free)")
APPROVAL = ("QC decides", "review when QC fails", "review every segment")
ACTIONS = ("run", "pause", "cancel", "resume", "approve held segment", "reject held segment",
           "generate alternate take",
           "retry segment (new seed)", "rollback to segment", "choose take")
DECISIONS = ("auto (QC + approval policy)", "accept", "reject (retry)", "hold for review")
SPEC = "h3.task_spec"
PLAN = "h3.segment_plan"
QC = "h3.qc_report"


def _jobs_dir() -> str:
    return str(job_folder("h3_jobs", create=True, who="H3 Job"))


def _sha(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


def _json(raw: str, contract: str, who: str) -> dict:
    try:
        d = json.loads(str(raw or ""))
    except ValueError as exc:
        raise ValueError(f"{who}: the input is not JSON ({exc}).") from exc
    if d.get("c2c_contract") != contract:
        raise ValueError(f"{who}: expected a {contract} contract, got {d.get('c2c_contract')!r}.")
    return d


# ── H3 Task Spec ───────────────────────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_TaskSpec(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TaskSpec", display_name="H3 Task Spec", category="MiniMax H3/Job",
            description="What a job does: the task, the frame range, segment length / context / blend / tracking "
                        "handles, seed, retries, the QC and approval policy, auto-continue. Fingerprints the source "
                        "and references, so a changed source makes only the segments that used it stale.",
            inputs=[
                io.Combo.Input("task", options=list(TASKS), default=TASKS[0]),
                io.Int.Input("first", default=-1, min=-1, max=10_000_000,
                             tooltip="First GLOBAL frame number of the job. -1 = the media's first frame number."),
                io.Int.Input("count", default=0, min=0, max=10_000_000, tooltip="Frames. 0 = to the media's end."),
                io.Int.Input("segment_frames", default=124, min=5, max=1000,
                             tooltip="Frames per segment, on the H3 grid (17k + 5: 22, 39, 56 ... 124, 141)."),
                io.Int.Input("context", default=39, min=0, max=500,
                             tooltip="Accepted frames each next segment continues from (H3 Continuation: 5, 22, 39 "
                                     "or 56)."),
                io.Int.Input("blend", default=0, min=0, max=200, tooltip="Frames generated twice and cross-faded."),
                io.Int.Input("lookbehind", default=0, min=0, max=500, tooltip="Tracking handles before (read only)."),
                io.Int.Input("lookahead", default=0, min=0, max=500, tooltip="Tracking handles after (read only)."),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip="The job's base seed: segment k's first attempt uses seed + k, retries add 1000."),
                io.Int.Input("max_retries", default=2, min=0, max=20),
                io.Combo.Input("approval", options=list(APPROVAL), default=APPROVAL[0],
                               tooltip="QC decides: a failed segment is retried with a new seed (up to max_retries, "
                                       "then it waits for review). review when QC fails: a failed segment waits for "
                                       "you at once. review every segment: every segment waits for you."),
                io.Boolean.Input("auto_continue", default=True,
                                 tooltip="After each committed segment, queue the next run automatically."),
                io.Int.Input("max_auto_runs", default=100, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0),
                io.Combo.Input("deliver_codec", options=["h264", "prores", "ffv1"], default="prores"),
                C2C_VIDEO.Input("media", optional=True, tooltip="The source plate (edit tasks)."),
                io.String.Input("prompt", default="", optional=True, multiline=True),
                io.String.Input("references", default="", optional=True,
                                tooltip="Reference identity (any text / JSON that changes when they change)."),
            ],
            outputs=[io.String.Output("task_spec", tooltip="h3.task_spec v1")],
        )

    @classmethod
    def execute(cls, task=TASKS[0], first=-1, count=0, segment_frames=124, context=39, blend=0, lookbehind=0,
                lookahead=0, seed=0, max_retries=2, approval=APPROVAL[0], auto_continue=True, max_auto_runs=100,
                fps=24.0, deliver_codec="prores", media=None, prompt="", references="") -> io.NodeOutput:
        fps_ = Fraction(fps).limit_denominator(1001000)
        fingerprints = {"prompt": _sha(prompt), "references": _sha(references)}
        media_first, media_frames = 0, None
        if media is not None:
            if not is_handle(media):
                raise ValueError("H3 Task Spec: 'media' is not a C2C video handle.")
            fingerprints["plate"] = media.fingerprint()
            nums = _media.source_numbers(media)
            sel = list(media.selected_indices())
            media_first = int(nums[sel[0]]) if sel and sel[0] >= 0 else 0
            media_frames = len(media)
            fps_ = Fraction(media.fps)
        start = media_first if int(first) < 0 else int(first)
        n = int(count) or (media_frames - (start - media_first) if media_frames is not None else 0)
        if n <= 0:
            raise ValueError("H3 Task Spec: 'count' is 0 and there is no media to take the length from.")
        spec = {"c2c_contract": SPEC, "c2c_contract_version": "1.0", "task": task, "first": start, "count": n,
                "media_first": media_first, "segment_frames": int(segment_frames), "context": int(context),
                "blend": int(blend), "lookbehind": int(lookbehind), "lookahead": int(lookahead), "seed": int(seed),
                "max_retries": int(max_retries), "approval": approval, "auto_continue": bool(auto_continue),
                "max_auto_runs": int(max_auto_runs), "fps": f"{fps_.numerator}/{fps_.denominator}",
                "deliver_codec": deliver_codec, "fingerprints": fingerprints, "prompt": prompt}
        return io.NodeOutput(json.dumps(spec))


# ── H3 Compatibility Validator ─────────────────────────────────────────────────────────────────────────────────────

def validate_spec(spec: dict, media=None, has_model: bool | None = None, has_vae: bool | None = None) -> dict:
    """{ok, errors: [...], warnings: [...]} - Rev 3 / H3 rules in words."""
    errors, warnings = [], []
    L_ = int(spec["segment_frames"])
    if HC.align_frame_count(L_) != L_:
        errors.append(f"segment_frames {L_} is not on the H3 grid (17k + 5): use {HC.align_frame_count(L_)}.")
    if spec["context"] + spec["blend"] >= L_:
        errors.append(f"context + blend ({spec['context'] + spec['blend']}) must be shorter than a segment ({L_}).")
    if spec["context"] and spec["context"] not in (5, 22, 39, 56):
        warnings.append(f"context {spec['context']}: H3 Continuation continues from 5, 22, 39 or 56 frames (whole "
                        f"latent steps); other values need a non-latent continuation.")
    fps = Fraction(spec["fps"])
    if fps != HC.FPS:
        warnings.append(f"the rate is {float(fps):g} fps; H3 generates at {HC.FPS} - conform first (H3 Media Conform).")
    if media is not None:
        w, h = int(media.width), int(media.height)
        if w % HC.CANVAS_MULTIPLE or h % HC.CANVAS_MULTIPLE:
            errors.append(f"the plate is {w}x{h}; H3 needs multiples of {HC.CANVAS_MULTIPLE} - conform the size "
                          f"(H3 Media Conform, multiple_of {HC.CANVAS_MULTIPLE}).")
        lo = spec["first"] - spec["media_first"]
        if lo < 0 or lo + spec["count"] > len(media):
            errors.append(f"frames {spec['first']}-{spec['first'] + spec['count'] - 1} reach outside the media "
                          f"({spec['media_first']}-{spec['media_first'] + len(media) - 1}).")
    elif spec["task"].startswith("edit"):
        errors.append("an edit task needs its source media - connect it to H3 Task Spec.")
    if has_model is False:
        warnings.append("no model connected to the validator - the graph must load an H3 model before it samples.")
    if has_vae is False:
        warnings.append("no VAE connected to the validator.")
    return {"c2c_contract": "h3.validation", "c2c_contract_version": "1.0", "ok": not errors, "errors": errors,
            "warnings": warnings}


class MiniMaxH3_CompatibilityValidator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_CompatibilityValidator", display_name="H3 Compatibility Validator",
            category="MiniMax H3/Job",
            description="Checks a task spec against its inputs before anything runs: the 17k + 5 grid, sizes in "
                        "multiples of 32, 24 fps, the frame range inside the media, the context length. strict "
                        "stops the run with the reasons; otherwise they are reported.",
            inputs=[io.String.Input("task_spec", force_input=True), C2C_VIDEO.Input("media", optional=True),
                    io.Model.Input("model", optional=True), io.Vae.Input("vae", optional=True),
                    io.Boolean.Input("strict", default=True)],
            outputs=[io.String.Output("task_spec", tooltip="Passed through when it is valid."),
                     io.String.Output("report"), io.Boolean.Output("ok")],
        )

    @classmethod
    def execute(cls, task_spec, media=None, model=None, vae=None, strict=True) -> io.NodeOutput:
        spec = _json(task_spec, SPEC, "H3 Compatibility Validator")
        rep = validate_spec(spec, media, None if model is None else True, None if vae is None else True)
        if strict and not rep["ok"]:
            raise ValueError("H3 Compatibility Validator: " + " ".join(rep["errors"]))
        return io.NodeOutput(task_spec, json.dumps(rep), bool(rep["ok"]))


# ── H3 Segment Planner ─────────────────────────────────────────────────────────────────────────────────────────────

def plan_from_spec(spec: dict) -> dict:
    segs = J.plan_segments(spec["first"], spec["count"], segment_frames=spec["segment_frames"],
                           context=spec["context"], blend=spec["blend"], lookbehind=spec["lookbehind"],
                           lookahead=spec["lookahead"])
    led = L.new(list(range(spec["count"])), fps=spec["fps"], offset=spec["first"],
                total_source=spec["count"])
    led = L.add_segments(led, [{"id": s["id"], "global": s["global"], "pad_head": s["pad_head"],
                                "pad_tail": s["pad_tail"], "overlap_head": s["overlap_head"],
                                "overlap_tail": s["overlap_tail"], "context_frames": s["context_frames"],
                                "accepted": s["accepted"]} for s in segs])
    return {"c2c_contract": PLAN, "c2c_contract_version": "1.0", "segments": segs, "ledger": led,
            "problems": J.check_tiling(segs, spec["first"], spec["count"]) + L.validate(led)}


class MiniMaxH3_SegmentPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SegmentPlanner", display_name="H3 Segment Planner", category="MiniMax H3/Job",
            description="Splits the job's range into H3 segments. Context, blend, tracking handles and grid padding "
                        "stay separate fields; the accepted ranges tile the range exactly (checked).",
            inputs=[io.String.Input("task_spec", force_input=True)],
            outputs=[io.String.Output("segment_plan", tooltip="h3.segment_plan v1 (with an h3.frame_ledger)."),
                     io.String.Output("table"), io.Int.Output("segments")],
        )

    @classmethod
    def execute(cls, task_spec) -> io.NodeOutput:
        plan = plan_from_spec(_json(task_spec, SPEC, "H3 Segment Planner"))
        if plan["problems"]:
            raise ValueError("H3 Segment Planner: " + "; ".join(plan["problems"]))
        rows = [f"seg {s['id']:>3}  model {s['global'][0]}-{s['global'][1] - 1} (+{s['pad_tail']} pad)  "
                f"context {s['context_frames']}  accept {s['accepted'][0]}-{s['accepted'][1] - 1}"
                for s in plan["segments"]]
        return io.NodeOutput(json.dumps(plan), "\n".join(rows), len(plan["segments"]))


# ── H3 Job Controller ──────────────────────────────────────────────────────────────────────────────────────────────

def _blocked():
    from comfy_execution.graph_utils import ExecutionBlocker
    return ExecutionBlocker(None)


def apply_action(job: J.Job, action: str, seg_n: int, take: str) -> str:
    """A user action on the job, before this run picks its segment. Returns a note."""
    if action == "pause":
        job.pause()
        return "paused"
    if action == "cancel":
        job.cancel()
        return "cancelled"
    if action == "resume":
        job.resume()
        return "resumed"
    if action in ("approve held segment", "reject held segment"):
        held = [p["id"] for p in job.m["plan"] if job.seg(p["id"])["status"] == "waiting review"]
        if not held:
            return "nothing is waiting for review"
        sid = held[0]
        if action.startswith("approve"):
            att = job.seg(sid)["attempts"][-1]
            job.accept(sid, {k: v.get("path") if isinstance(v, dict) else v
                             for k, v in (att.get("outputs") or {}).items()}, note="approved by the user")
            return f"segment {sid} approved"
        job.seg(sid)["status"] = "rejected"
        job.transition("GENERATING", f"segment {sid} rejected by the user")
        return f"segment {sid} rejected - generating it again"
    if action == "retry segment (new seed)":
        job.rollback(seg_n - 1 if seg_n > 0 else None)          # it and the segments continuing from it
        return f"segment {seg_n} (and the segments after it) will be generated again with a new seed"
    if action == "rollback to segment":
        gone = job.rollback(seg_n)
        return f"rolled back to segment {seg_n} ({len(gone)} later segment(s) superseded)"
    if action == "choose take":
        stale = job.choose_take(seg_n, take)
        return f"segment {seg_n} uses take {take}; {len(stale)} later segment(s) to redo"
    if job.state == "PAUSED":
        job.resume("run")
    return ""


class MiniMaxH3_JobController(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_JobController", display_name="H3 Job Controller", category="MiniMax H3/Job",
            description="Opens (or creates) the job, applies an action, and hands THIS run the next segment: its "
                        "source window, seed and ranges. With auto-continue the Commit queues the next run itself - "
                        "one click runs the whole job. Nothing left to do: the generation is skipped.",
            inputs=[
                io.String.Input("task_spec", force_input=True),
                io.String.Input("segment_plan", force_input=True),
                io.String.Input("job", default="h3_job"),
                io.Combo.Input("action", options=list(ACTIONS), default="run"),
                io.Int.Input("action_segment", default=0, min=0, max=100000),
                io.String.Input("action_take", default=""),
                C2C_VIDEO.Input("media", optional=True, tooltip="The source plate (edit tasks)."),
            ],
            outputs=[
                C2C_VIDEO.Output("window", tooltip="The plate frames this segment generates over (edit tasks)."),
                C2C_VIDEO.Output("track_window", tooltip="The same, with the tracking handles."),
                io.Image.Output("context_images", tooltip="The accepted frames this segment continues from."),
                io.Int.Output("seed"),
                io.Int.Output("frames", tooltip="Model frames of this segment (17k + 5)."),
                io.String.Output("segment", tooltip="Feed H3 QC Gate and H3 Job Commit."),
                io.String.Output("status"),
            ],
            # the auto-continue token rides in the queued run's extra_pnginfo: no widget for anyone to edit
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def fingerprint_inputs(cls, task_spec, segment_plan, job="h3_job", action="run", action_segment=0,
                           action_take="", media=None):
        try:
            return J.Job.open(_jobs_dir(), job, create=False, recover=False).state_token()
        except Exception:  # noqa: BLE001 - a new job: run
            return "new"

    @classmethod
    def execute(cls, task_spec, segment_plan, job="h3_job", action="run", action_segment=0, action_take="",
                media=None) -> io.NodeOutput:
        auto_token = str(((getattr(cls.hidden, "extra_pnginfo", None) or {}).get("h3_auto_token")) or "")
        spec = _json(task_spec, SPEC, "H3 Job Controller")
        plan = _json(segment_plan, PLAN, "H3 Job Controller")
        j = J.Job.open(_jobs_dir(), job, spec=spec)
        if not j.m["plan"] or [s["accepted"] for s in j.m["plan"]] != [s["accepted"] for s in plan["segments"]]:
            j.set_plan(plan["segments"])
        auto = bool(auto_token) and auto_token == j.m.get("auto_token")
        if not auto:
            j.m["auto_chain"] = 0                                   # a run a person queued starts a new chain
            j.save()
        note = apply_action(j, action, int(action_segment), str(action_take)) if not auto else ""
        if j.state in J._SETUP:
            j.transition("GENERATING", "start")
        if j.state == "ASSEMBLING":
            # every segment is accepted but the run that assembled died (crash, interrupt): finish it here - the Commit
            # will not run again, this run has no segment to hand it
            final = assemble(j, j.m["spec"])
            j.complete({"video": final})
            note = (note + "; " if note else "") + f"assembled {final}"
        alternate = ""
        if action == "generate alternate take" and not auto:
            # an extra take of an ACCEPTED segment, kept beside the chosen one (choose it later with "choose take")
            seg = j.seg(int(action_segment))
            if seg["status"] != "accepted":
                raise ValueError(f"H3 Job Controller: segment {action_segment} is {seg['status']} - an alternate "
                                 f"take needs an accepted segment to sit beside.")
            alternate = str(action_take or "").strip() or f"take{len(seg['attempts']) + 1}"
        else:
            seg = j.next_segment() if j.state not in J.STOPPED else None
        if not alternate and (seg is None or j.state in J.STOPPED):
            why = {"PAUSED": "paused - set action to resume or run", "CANCELLED": "cancelled",
                   "WAITING_FOR_REVIEW": "a segment waits for review - approve or reject it (action)",
                   "COMPLETED": "completed", "FAILED": "failed - see the job history",
                   "ASSEMBLING": "every segment is accepted - the Commit assembles"}.get(j.state, j.state)
            status = f"job {j.m['name']}: {why}" + (f" ({note})" if note else "")
            b = _blocked()
            return io.NodeOutput(b, b, b, b, b, json.dumps({"job": j.m["name"], "state": j.state, "idle": True}),
                                 status)
        sid = int(seg["id"])
        ok, why = j.context_ok(sid)
        if not ok:
            j.hold_for_review(sid, why)
            b = _blocked()
            return io.NodeOutput(b, b, b, b, b, json.dumps({"job": j.m["name"], "state": j.state, "idle": True}),
                                 f"job {j.m['name']}: {why}")
        p = j.plan_of(sid)
        policy = "new seed" if seg["attempts"] else "first"
        seed = j.seed_for(sid, "new seed")
        att = j.begin(sid, seed=seed, inputs={"uses": ["plate", "prompt", "references"]},
                      changes="alternate take" if alternate else ("" if policy == "first" else "retry"),
                      alternate=bool(alternate))
        win = track = None
        if media is not None and is_handle(media):
            mf = int(spec["media_first"])
            g0, g1 = p["global"]
            win = media.trimmed(max(0, g0 - mf), max(0, g1 - mf))
            t0, t1 = p["track_window"]
            track = media.trimmed(max(0, t0 - mf), max(0, t1 - mf))
        ctx = _context_frames(j, p) if p["context_frames"] else torch.zeros((1, 8, 8, 3))
        seg_json = {"job": j.m["name"], "job_path": j.path, "segment": sid, "attempt": att["n"], "seed": seed,
                    "plan": p, "controller": str(getattr(cls.hidden, "unique_id", None) or ""), "approval": spec["approval"],
                    "max_retries": spec["max_retries"], "alternate": alternate}
        status = (f"job {j.m['name']}: segment {sid + 1} of {len(j.m['plan'])}, attempt {att['n']}, seed {seed}"
                  + (f" - alternate take '{alternate}'" if alternate else "")
                  + (f" ({note})" if note else ""))
        b = _blocked()
        return io.NodeOutput(win if win is not None else b, track if track is not None else b, ctx, seed,
                             int(p["model_frames"]), json.dumps(seg_json), status)


def _attempt_dir(j: J.Job, sid: int, n: int) -> str:
    return os.path.join(j.root, "segments", f"seg_{sid:04d}_a{n:02d}")


def _context_frames(j: J.Job, p: dict) -> torch.Tensor:
    """The last ``context_frames`` accepted frames before this segment (from the context segment's chosen take)."""
    cf = p.get("context_from")
    if cf is None:
        return torch.zeros((1, 8, 8, 3))
    s = j.seg(cf)
    n = s["takes"].get(s.get("take") or "main", len(s["attempts"]))
    d = _attempt_dir(j, int(cf), int(n))
    frames = sorted(f for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else []
    want = frames[-int(p["context_frames"]):]
    if not want:
        return torch.zeros((1, 8, 8, 3))
    return torch.stack([_read_png16(os.path.join(d, f)) for f in want])


def _write_png16(path: str, frame: torch.Tensor) -> None:
    import cv2
    arr = np.clip(frame[..., :3].float().cpu().numpy(), 0.0, 1.0)
    cv2.imwrite(path, np.rint(arr[..., ::-1] * 65535.0).astype(np.uint16))


def _read_png16(path: str) -> torch.Tensor:
    import cv2
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = arr[..., ::-1].astype(np.float32) / (65535.0 if arr.dtype == np.uint16 else 255.0)
    return torch.from_numpy(np.ascontiguousarray(rgb))


# ── H3 QC Gate ─────────────────────────────────────────────────────────────────────────────────────────────────────

def qc_checks(images: torch.Tensor, seg: dict, context: torch.Tensor | None = None, *, min_mean: float = 0.01,
              freeze_max: float = 0.5, seam_max: float = 0.08, diff_report: dict | None = None,
              diff_max: float = 0.0) -> dict:
    p = seg["plan"]
    checks = []
    expected = int(p["model_frames"])
    checks.append({"name": "frame count", "ok": int(images.shape[0]) >= expected - int(p["pad_tail"]),
                   "value": int(images.shape[0]), "expected": expected})
    finite = bool(torch.isfinite(images).all())
    checks.append({"name": "finite values", "ok": finite})
    means = images[..., :3].float().mean(dim=(1, 2, 3))
    dark = [int(i) for i in torch.nonzero(means < min_mean).flatten().tolist()]
    checks.append({"name": "not black", "ok": len(dark) <= max(0, int(0.05 * len(means))), "frames": dark[:50]})
    if images.shape[0] > 1:
        step = (images[1:, ..., :3] - images[:-1, ..., :3]).abs().mean(dim=(1, 2, 3))
        frozen = float((step < 1e-4).float().mean())
        checks.append({"name": "not frozen", "ok": frozen <= freeze_max, "value": round(frozen, 4)})
    if context is not None and int(p["context_frames"]) and context.shape[0] == int(p["context_frames"]) \
            and context.shape[1:3] == images.shape[1:3]:
        seam = float((images[: context.shape[0], ..., :3] - context[..., :3]).abs().mean())
        checks.append({"name": "continues the accepted frames", "ok": seam <= seam_max, "value": round(seam, 4)})
    if diff_report and diff_max > 0:
        checks.append({"name": "difference from the source", "ok": float(diff_report.get("max", 0)) <= diff_max,
                       "value": diff_report.get("max")})
    verdict = "pass" if all(c["ok"] for c in checks) else "fail"
    return {"c2c_contract": QC, "c2c_contract_version": "1.0", "segment": seg["segment"], "attempt": seg["attempt"],
            "verdict": verdict, "checks": checks}


class MiniMaxH3_QCGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_QCGate", display_name="H3 QC Gate", category="MiniMax H3/Job",
            description="Judges the generated segment: frame count, finite values, no black frames, not frozen, "
                        "continues the accepted frames at the context, optionally the difference from the source (an "
                        "H3 Sequence Diff report). The verdict decides commit / retry / review in H3 Job Commit.",
            inputs=[io.String.Input("segment", force_input=True), io.Image.Input("images"),
                    io.Image.Input("context_images", optional=True),
                    io.String.Input("diff_report", default="", optional=True, force_input=True),
                    io.Float.Input("min_mean", default=0.01, min=0.0, max=1.0, step=0.001),
                    io.Float.Input("freeze_max", default=0.5, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("seam_max", default=0.08, min=0.0, max=1.0, step=0.001),
                    io.Float.Input("diff_max", default=0.0, min=0.0, max=1.0, step=0.001,
                                   tooltip="0 = do not judge the difference report.")],
            outputs=[io.String.Output("qc_report"), io.Boolean.Output("passed"), io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, segment, images, context_images=None, diff_report="", min_mean=0.01, freeze_max=0.5,
                seam_max=0.08, diff_max=0.0) -> io.NodeOutput:
        seg = json.loads(segment)
        dr = json.loads(diff_report) if str(diff_report or "").strip() else None
        rep = qc_checks(images, seg, context_images, min_mean=min_mean, freeze_max=freeze_max, seam_max=seam_max,
                        diff_report=dr, diff_max=diff_max)
        j = J.Job.open(os.path.dirname(os.path.dirname(seg["job_path"])), seg["job"], create=False, recover=False)
        j.record_qc(int(seg["segment"]), rep)
        return io.NodeOutput(json.dumps(rep), rep["verdict"] == "pass", images)


# ── H3 Job Commit (+ auto-continue) ────────────────────────────────────────────────────────────────────────────────

def _server_url() -> str:
    try:
        from comfy.cli_args import args  # type: ignore[import-not-found]
        port = int(getattr(args, "port", 8188))
        hosts = [h.strip() for h in str(getattr(args, "listen", "127.0.0.1") or "127.0.0.1").split(",") if h.strip()]
        host = "127.0.0.1" if any(h in ("0.0.0.0", "::", "127.0.0.1", "localhost") for h in hosts) else hosts[0]
        scheme = "https" if getattr(args, "tls_keyfile", None) else "http"
    except Exception:  # noqa: BLE001
        port, host, scheme = 8188, "127.0.0.1", "http"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def post_prompt(prompt: dict, extra_pnginfo: dict | None) -> str:
    """Queue ``prompt`` through the server's own public POST /prompt (validation, ids, on_prompt hooks all apply).
    ``extra_pnginfo`` carries the workflow (for saved files) and the auto-continue token."""
    import ssl
    import urllib.request

    body = json.dumps({"prompt": prompt, "extra_data": {"extra_pnginfo": extra_pnginfo or {}}}).encode("utf-8")
    req = urllib.request.Request(_server_url() + "/prompt", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    ctx = ssl._create_unverified_context() if req.full_url.startswith("https") else None   # our own server
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return json.loads(r.read() or b"{}").get("prompt_id", "")


_POSTER = post_prompt                                          # tests replace it


def auto_continue(j: J.Job, spec: dict, seg: dict, prompt: dict | None, extra_pnginfo) -> str:
    if os.environ.get("H3_AUTO_CONTINUE", "").strip() == "0":
        return "auto-continue is switched off (H3_AUTO_CONTINUE=0)"
    if not spec.get("auto_continue"):
        return "auto-continue is off in the task spec - queue the next run yourself"
    if j.state in J.STOPPED or j.state in ("ASSEMBLING",) or j.next_segment() is None:
        return ""
    chain = int(j.m.get("auto_chain", 0))
    if chain >= int(spec.get("max_auto_runs", 100)):
        return f"auto-continue stopped after {chain} runs (max_auto_runs) - queue again to continue"
    if not isinstance(prompt, dict) or not seg.get("controller") or seg["controller"] not in prompt:
        return "auto-continue could not find the Job Controller in this prompt - queue the next run yourself"
    token = uuid.uuid4().hex
    nxt = json.loads(json.dumps(prompt))
    nxt[seg["controller"]].setdefault("inputs", {})["action"] = "run"     # an approve / retry is not repeated
    extra = dict(extra_pnginfo or {}, h3_auto_token=token)
    j.m["auto_token"], j.m["auto_chain"] = token, chain + 1
    j.m["auto_runs"] = int(j.m.get("auto_runs", 0)) + 1
    j.save()
    try:
        pid = _POSTER(nxt, extra)
    except Exception as exc:  # noqa: BLE001 - said in the status, never retried in a loop
        return f"auto-continue could not queue the next run ({exc}) - queue it yourself"
    return f"queued the next segment automatically (run {chain + 1}, prompt {pid[:8]})"


def assemble(j: J.Job, spec: dict) -> str:
    """The final movie from every segment's chosen take, in plan order, streamed."""
    codec = spec.get("deliver_codec", "prores")
    ext = {"h264": ".mp4", "prores": ".mov", "ffv1": ".mkv"}[codec]
    out = os.path.join(j.root, f"{j.m['name']}{ext}")
    files: list[str] = []
    for p in j.m["plan"]:
        s = j.seg(p["id"])
        n = s["takes"].get(s.get("take") or "main", len(s["attempts"]))
        d = _attempt_dir(j, int(p["id"]), int(n))
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]
    if not files:
        raise ValueError("H3 Job Commit: nothing to assemble.")
    first = _read_png16(files[0])
    h, w = int(first.shape[0]), int(first.shape[1])
    if codec != "ffv1":
        w, h = w - w % 2, h - h % 2
    part = out + ".partial" + ext
    with StreamWriter(part, Fraction(spec["fps"]), w, h, codec=codec, ten_bit=codec != "h264") as wr:
        for i in range(0, len(files), 8):
            wr.write(torch.stack([_read_png16(f)[:h, :w] for f in files[i:i + 8]]))
    os.replace(part, out)
    return out


class MiniMaxH3_JobCommit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_JobCommit", display_name="H3 Job Commit", category="MiniMax H3/Job",
            description="Records the segment: accepted (its frames kept, numbered by global frame), retried with a "
                        "new seed, or held for review - by the QC verdict and the approval policy, or your "
                        "decision. When every segment is accepted it assembles the movie. With auto-continue it "
                        "queues the next run itself.",
            inputs=[io.String.Input("segment", force_input=True), io.Image.Input("images"),
                    io.String.Input("qc_report", default="", optional=True, force_input=True),
                    io.Combo.Input("decision", options=list(DECISIONS), default=DECISIONS[0])],
            outputs=[io.String.Output("status"), io.String.Output("job_json"), io.String.Output("final_path")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, segment, images, qc_report="", decision=DECISIONS[0]) -> io.NodeOutput:
        seg = json.loads(segment)
        j = J.Job.open(os.path.dirname(os.path.dirname(seg["job_path"])), seg["job"], create=False, recover=False)
        spec = j.m["spec"]
        sid, n, p = int(seg["segment"]), int(seg["attempt"]), seg["plan"]
        qc = json.loads(qc_report) if str(qc_report or "").strip() else None
        verdict = (qc or {}).get("verdict", "pass")
        approval = spec.get("approval", APPROVAL[0])
        if decision.startswith("auto"):
            if approval == "review every segment":
                choice = "review"
            elif verdict == "pass":
                choice = "accept"
            else:                                               # QC decides: retry (the engine caps it); else ask
                choice = "reject" if approval == "QC decides" else "review"
        else:
            choice = {"accept": "accept", "reject (retry)": "reject", "hold for review": "review"}[decision]
        notes = []
        if choice in ("accept", "review"):
            d = _attempt_dir(j, sid, n)
            os.makedirs(d, exist_ok=True)
            a0, a1 = p["accepted"]
            off = a0 - p["global"][0]
            keep = images[off: off + (a1 - a0)]
            if keep.shape[0] != a1 - a0:
                raise ValueError(f"H3 Job Commit: segment {sid} must cover frames {a0}-{a1 - 1}, the images give "
                                 f"{keep.shape[0]} of {a1 - a0} (generate {p['model_frames']} frames).")
            for k in range(keep.shape[0]):
                _write_png16(os.path.join(d, f"frame_{a0 + k:07d}.png"), keep[k])
            digest = hashlib.sha256("".join(J.sha256_file(os.path.join(d, f))
                                            for f in sorted(os.listdir(d))).encode()).hexdigest()
            outputs = {"frames_dir": d, "frames": int(keep.shape[0]), "frames_sha256": digest}
            if seg.get("alternate"):                            # kept as a named take; nothing else moves
                j.accept(sid, outputs, take=seg["alternate"])
                notes.append(f"segment {sid}: take '{seg['alternate']}' kept beside the chosen one")
            elif choice == "accept":
                j.accept(sid, outputs)
                notes.append(f"segment {sid} accepted (frames {a0}-{a1 - 1})")
            else:
                j.seg(sid)["attempts"][-1]["outputs"] = outputs
                j.hold_for_review(sid, f"QC {verdict}" if verdict != "pass" else "review every segment")
                notes.append(f"segment {sid} waits for review: {d}")
        else:
            res = j.reject(sid, f"QC {verdict}" if decision.startswith("auto") else "rejected",
                           max_retries=int(spec.get("max_retries", 2)))
            notes.append(f"segment {sid} rejected - " + ("retrying with a new seed" if res == "retry"
                                                          else "retries used up, waiting for review"))
        final = ""
        if j.state == "ASSEMBLING":
            final = assemble(j, spec)
            j.complete({"video": final})
            notes.append(f"job complete: {final}")
        msg = auto_continue(j, spec, seg, getattr(cls.hidden, "prompt", None),
                            getattr(cls.hidden, "extra_pnginfo", None))
        if msg:
            notes.append(msg)
        status = f"job {j.m['name']} [{j.state}]: " + "; ".join(notes)
        return io.NodeOutput(status, json.dumps(j.summary()), final, ui={"text": [status]})


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_TaskSpec": MiniMaxH3_TaskSpec,
    "MiniMaxH3_CompatibilityValidator": MiniMaxH3_CompatibilityValidator,
    "MiniMaxH3_SegmentPlanner": MiniMaxH3_SegmentPlanner,
    "MiniMaxH3_JobController": MiniMaxH3_JobController,
    "MiniMaxH3_QCGate": MiniMaxH3_QCGate,
    "MiniMaxH3_JobCommit": MiniMaxH3_JobCommit,
}
