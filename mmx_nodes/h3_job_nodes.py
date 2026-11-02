"""H3 job nodes: Task Spec, Compatibility Validator, Segment Planner, Job Controller, QC Gate, Job Commit (owner A21
Revision 3 Parts 7-8; ledger L9.75). On the engine in mmx_utils/h3_job.py.

One queue click runs a whole job: the Controller hands this run the next segment, the graph generates it, the QC
Gate judges it, and the Commit records it and - when more is left and nothing needs a person - posts the SAME prompt
to the server's own POST /prompt, so the next run takes the next segment (Rev 3 8.5 "users must not manually queue
every continuation segment"). Guards: the job's state (paused / review / completed stop it), a chain cap
(``max_auto_runs``), the kill switch ``H3_AUTO_CONTINUE=0``.

Accepted frames are kept as 16-bit PNGs per segment attempt (exact, any length), numbered by GLOBAL frame; the final
movie is assembled from the chosen takes in plan order, streamed (one chunk in memory). Frames generated past a
segment's accepted end (blend / dissolve tail) are kept beside them and cross-faded at assembly.

Long generative video (Rev 3 Parts 11 + 15; L9.77): H3 Shot Planner turns a prompt timeline into shots (cut /
continue / dissolve; context never crosses a cut), H3 Reference Bank keeps role-tagged references with duplicate
marks, H3 Subject Memory rebuilds the scene's identity anchors and colour baseline from ACCEPTED segments only.

Original nodes, GPL-3.0. Design: docs/evidence/A21/phase2/DESIGN_L9.75_job_engine.md and DESIGN_L9.77_long_video.md
in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_constants as HC  # noqa: E402
from mmx_utils import h3_job as J  # noqa: E402
from mmx_utils.h3_session import frame_stats  # noqa: E402
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
SHOTS = "h3.shot_plan"
BANK = "h3.reference_bank"
MEMORY = "h3.subject_memory"
ROLES = ("identity", "face", "head", "hair", "body", "outfit", "footwear", "props", "expression", "performance",
         "scene", "lighting", "camera", "style")
IDENTITY_ROLES = ("identity", "face", "head")
MEMORY_ROLES = ("identity", "face", "head", "hair", "body", "outfit", "props")


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
                                tooltip="Reference identity (any text / JSON that changes when they change) - e.g. "
                                        "H3 Reference Bank's fingerprint."),
                io.Int.Input("hold_head", default=0, min=0, max=64, optional=True,
                             tooltip="Frames of the first picture a segment without continuation context generates "
                                     "before its window and then drops (the PRB workflow's AV handles: 8). The model's "
                                     "first frames settle on a held picture, not on the cut."),
                io.String.Input("shot_plan", optional=True, force_input=True,
                                tooltip="From H3 Shot Planner: the job becomes its shots (their length sets 'count'; "
                                        "each segment gets its shot's prompt, with 'prompt' above as a shared "
                                        "prefix). Cuts never carry context; dissolves cross-fade."),
            ],
            outputs=[io.String.Output("task_spec", tooltip="h3.task_spec v1")],
        )

    @classmethod
    def execute(cls, task=TASKS[0], first=-1, count=0, segment_frames=124, context=39, blend=0, lookbehind=0,
                lookahead=0, seed=0, max_retries=2, approval=APPROVAL[0], auto_continue=True, max_auto_runs=100,
                fps=24.0, deliver_codec="prores", media=None, prompt="", references="", shot_plan=None,
                hold_head=0) -> io.NodeOutput:
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
        shots = None
        if str(shot_plan or "").strip():
            sp = _json(shot_plan, SHOTS, "H3 Task Spec")
            if Fraction(sp["fps"]) != fps_:
                raise ValueError(f"H3 Task Spec: the shot plan was timed at {float(Fraction(sp['fps'])):g} fps and the "
                                 f"job runs at {float(fps_):g} - set the same fps on H3 Shot Planner.")
            if int(count) and int(count) != int(sp["frames"]):
                raise ValueError(f"H3 Task Spec: 'count' is {int(count)} and the shot plan lasts {sp['frames']} "
                                 f"frames - set count to 0 (the shots decide) or change the shots.")
            count = int(sp["frames"])
            # the plan's frames are offsets from the job's first frame
            shots = [{**s, "global": [start + int(s["global"][0]), start + int(s["global"][1])]} for s in sp["shots"]]
        n = int(count) or (media_frames - (start - media_first) if media_frames is not None else 0)
        if n <= 0:
            raise ValueError("H3 Task Spec: 'count' is 0 and there is no media to take the length from.")
        spec = {"c2c_contract": SPEC, "c2c_contract_version": "1.0", "task": task, "first": start, "count": n,
                "media_first": media_first, "segment_frames": int(segment_frames), "context": int(context),
                "blend": int(blend), "lookbehind": int(lookbehind), "lookahead": int(lookahead), "seed": int(seed),
                "max_retries": int(max_retries), "approval": approval, "auto_continue": bool(auto_continue),
                "max_auto_runs": int(max_auto_runs), "fps": f"{fps_.numerator}/{fps_.denominator}",
                "deliver_codec": deliver_codec, "fingerprints": fingerprints, "prompt": prompt}
        if shots is not None:
            # not a fingerprint: an edited shot re-opens only its own segments (Job.set_plan compares plan entries)
            spec["shots"] = shots
        spec["hold_head"] = int(hold_head or 0)
        if media is not None:
            audio = audio_span(media.trimmed(start - media_first, start - media_first + n))
            if audio:
                spec["audio"] = audio                           # the assembly puts the plate's sound back
        return io.NodeOutput(json.dumps(spec))


def audio_span(handle) -> dict | None:
    """Where the plate's sound for these frames is (file, stream, start, length), for the assembly to mux. None for
    image sequences and silent clips. Read from the vendored lazy-audio object the loader itself uses."""
    from mmx_vendor.c2c_core.video.audio import lazy_audio_for_handle
    la = lazy_audio_for_handle(handle)
    path = getattr(la, "_path", None)
    if not path:
        return None
    return {"path": str(path), "stream_index": int(getattr(la, "_stream_index", 0)),
            "start_sec": float(getattr(la, "_start_sec", 0.0)), "duration_sec": float(getattr(la, "_duration_sec", 0.0)),
            "reverse": bool(getattr(la, "_reverse", False))}


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
    kw = dict(segment_frames=spec["segment_frames"], context=spec["context"], blend=spec["blend"],
              lookbehind=spec["lookbehind"], lookahead=spec["lookahead"], hold_head=int(spec.get("hold_head", 0)))
    try:
        segs = J.plan_shots(spec["shots"], **kw) if spec.get("shots") else \
            J.plan_segments(spec["first"], spec["count"], **kw)
    except J.JobError as exc:
        raise ValueError(f"H3 Segment Planner: {exc}") from exc
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
        rows = [(f"shot {s['shot'] + 1:>2} {s['transition']:<8} " if "shot" in s else "")
                + f"seg {s['id']:>3}  model " + (f"{s['pad_head']} held + " if s.get("pad_head") else "")
                + f"{s['global'][0]}-{s['global'][1] - 1} (+{s['pad_tail']} pad)  "
                f"context {s['context_frames']}  accept {s['accepted'][0]}-{s['accepted'][1] - 1}"
                + (f"  cross-fade {s['overlap_head']}" if s["overlap_head"] else "")
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
                C2C_VIDEO.Input("matte", optional=True,
                                tooltip="A matte clip frame-aligned with the plate: matte_window is cut exactly like "
                                        "window (same frames, same held frames)."),
            ],
            outputs=[
                C2C_VIDEO.Output("window", tooltip="The plate frames this segment generates over (edit tasks)."),
                C2C_VIDEO.Output("track_window", tooltip="The same, with the tracking handles."),
                io.Image.Output("context_images", tooltip="The accepted frames this segment continues from."),
                io.Int.Output("seed"),
                io.Int.Output("frames", tooltip="Model frames of this segment (17k + 5)."),
                io.String.Output("segment", tooltip="Feed H3 QC Gate and H3 Job Commit."),
                io.String.Output("status"),
                io.String.Output("prompt", tooltip="This segment's prompt: the task spec's prompt, then its shot's "
                                                   "prompt when the job has a shot plan."),
                io.Latent.Output("context_latent",
                                 tooltip="The accepted latent of the segment this one continues from (its chosen "
                                         "take), for H3 Motion Context's context_latent. Empty when there is none."),
                C2C_VIDEO.Output("matte_window", tooltip="The matte, cut exactly like window."),
            ],
            # the auto-continue token rides in the queued run's extra_pnginfo: no widget for anyone to edit
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def fingerprint_inputs(cls, task_spec, segment_plan, job="h3_job", action="run", action_segment=0,
                           action_take="", media=None, matte=None):
        try:
            return J.Job.open(_jobs_dir(), job, create=False, recover=False).state_token()
        except Exception:  # noqa: BLE001 - a new job: run
            return "new"

    @classmethod
    def execute(cls, task_spec, segment_plan, job="h3_job", action="run", action_segment=0, action_take="",
                media=None, matte=None) -> io.NodeOutput:
        auto_token = str(((getattr(cls.hidden, "extra_pnginfo", None) or {}).get("h3_auto_token")) or "")
        spec = _json(task_spec, SPEC, "H3 Job Controller")
        plan = _json(segment_plan, PLAN, "H3 Job Controller")
        j = J.Job.open(_jobs_dir(), job, spec=spec)
        if not j.m["plan"] or [J.plan_key(s) for s in j.m["plan"]] != [J.plan_key(s) for s in plan["segments"]]:
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
                                 status, b, b, b)
        sid = int(seg["id"])
        ok, why = j.context_ok(sid)
        if not ok:
            j.hold_for_review(sid, why)
            b = _blocked()
            return io.NodeOutput(b, b, b, b, b, json.dumps({"job": j.m["name"], "state": j.state, "idle": True}),
                                 f"job {j.m['name']}: {why}", b, b, b)
        p = j.plan_of(sid)
        policy = "new seed" if seg["attempts"] else "first"
        seed = j.seed_for(sid, "new seed")
        att = j.begin(sid, seed=seed, inputs={"uses": ["plate", "prompt", "references"]},
                      changes="alternate take" if alternate else ("" if policy == "first" else "retry"),
                      alternate=bool(alternate))
        win = track = mwin = None
        if media is not None and is_handle(media):
            mf = int(spec["media_first"])
            g0, g1 = p["global"]
            win = generation_window(media, g0 - mf, g1 - mf, int(p.get("pad_head") or 0), int(p.get("pad_tail") or 0))
            t0, t1 = p["track_window"]
            track = media.trimmed(max(0, t0 - mf), max(0, t1 - mf))
            if matte is not None:
                if not is_handle(matte):
                    raise ValueError("H3 Job Controller: 'matte' is not a C2C video handle.")
                if (int(matte.width), int(matte.height)) != (int(media.width), int(media.height)):
                    raise ValueError(f"H3 Job Controller: the matte is {matte.width}x{matte.height} and the plate "
                                     f"{media.width}x{media.height} - they must be the same size, frame for frame.")
                if len(matte) < g1 - mf:
                    raise ValueError(f"H3 Job Controller: the matte has {len(matte)} frames; this segment needs the "
                                     f"plate's frames up to {g1 - mf} - give a matte as long as the plate.")
                mwin = generation_window(matte, g0 - mf, g1 - mf, int(p.get("pad_head") or 0), int(p.get("pad_tail") or 0))
        ctx = _context_frames(j, p) if p["context_frames"] else torch.zeros((1, 8, 8, 3))
        text = segment_prompt(spec, p)
        seg_json = {"job": j.m["name"], "job_path": j.path, "segment": sid, "attempt": att["n"], "seed": seed,
                    "plan": p, "controller": str(getattr(cls.hidden, "unique_id", None) or ""), "approval": spec["approval"],
                    "max_retries": spec["max_retries"], "alternate": alternate, "shot": p.get("shot"),
                    "scene": p.get("scene"), "transition": p.get("transition"), "prompt": text}
        status = (f"job {j.m['name']}: segment {sid + 1} of {len(j.m['plan'])}, attempt {att['n']}, seed {seed}"
                  + (f", shot {p['shot'] + 1} ({p['transition']})" if p.get("shot") is not None else "")
                  + (f" - alternate take '{alternate}'" if alternate else "")
                  + (f" ({note})" if note else ""))
        b = _blocked()
        return io.NodeOutput(win if win is not None else b, track if track is not None else b, ctx, seed,
                             int(p["model_frames"]), json.dumps(seg_json), status, text, _context_latent(j, p),
                             mwin if mwin is not None else b)


def generation_window(handle, lo: int, hi: int, hold: int, pad_tail: int = 0):
    """The model's frames for a segment: positions lo..hi of a handle, ``hold`` copies of its first frame before them
    (a timeline repeat: the lazy audio starts that many frames earlier with silence), and ``pad_tail`` grid-padding
    frames after them - the plate's own next frames while it has them, then copies of the last one - so the window is
    exactly the segment's model length (H3 Source AV Latent refuses anything off the 17k + 5 grid)."""
    hi_real = min(len(handle), max(0, hi) + max(0, int(pad_tail)))
    win = handle.trimmed(max(0, lo), hi_real)
    sel = [int(i) for i in win.selected_indices()]
    short = max(0, hi + int(pad_tail) - hi_real)
    if (hold > 0 or short) and sel:
        win = win.with_timeline([sel[0]] * max(0, hold) + sel + [sel[-1]] * short)
    return win


def segment_prompt(spec: dict, p: dict) -> str:
    """The task spec's prompt (shared by every shot), then the segment's shot prompt."""
    return ", ".join(x for x in (str(spec.get("prompt") or "").strip(), str(p.get("prompt") or "").strip()) if x)


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


LATENT_FILE = "latent.safetensors"


def save_latent(d: str, latent) -> str:
    """The accepted attempt's latent next to its frames: video (+ audio) streams, as H3 Motion Context saves them."""
    from mmx_utils.motion_context import streams_from_latent, write_safetensors
    samples = latent["samples"]
    # a plain tensor is ONE video stream (unbinding it would split the batch); an AV latent is a nested pair
    parts = [samples] if isinstance(samples, torch.Tensor) else streams_from_latent(latent)
    data = {"video": parts[0].cpu().contiguous()}
    if len(parts) > 1:
        data["audio"] = parts[1].cpu().contiguous()
    path = os.path.join(d, LATENT_FILE)
    write_safetensors(path, data)
    return path


def _context_latent(j: J.Job, p: dict):
    """The chosen take's latent of the segment this one continues from; None (= not connected) when there is none."""
    cf = p.get("context_from")
    if cf is None:
        return None
    s = j.seg(cf)
    path = os.path.join(_attempt_dir(j, int(cf), int(s["takes"].get(s.get("take") or "main", len(s["attempts"])))),
                        LATENT_FILE)
    if not os.path.isfile(path):
        return None
    from safetensors.torch import load_file
    data = load_file(path)
    video = data["video"].contiguous().clone()
    return {"samples": [video, data["audio"].contiguous().clone()]} if "audio" in data else {"samples": video}


def accepted_slice(p: dict) -> tuple[int, int]:
    """Where the accepted frames sit in a segment's generated images: after the held frames and the context."""
    a0, a1 = p["accepted"]
    return a0 - p["global"][0] + int(p.get("pad_head") or 0), a1 - a0


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
              diff_max: float = 0.0, memory: dict | None = None, colour_drift_max: float = 0.0,
              plate: torch.Tensor | None = None, mask: torch.Tensor | None = None, protect_px: int = 4,
              background_max: float = 0.008, mask_jump_max: float = 0.35, strict_masks: bool = False,
              seam_ratio_max: float = 1.5) -> dict:
    p = seg["plan"]
    checks = []
    expected = int(p["model_frames"])
    checks.append({"name": "frame count", "ok": int(images.shape[0]) >= expected - int(p["pad_tail"]),
                   "value": int(images.shape[0]), "expected": expected})
    finite = bool(torch.isfinite(images).all())
    checks.append({"name": "finite values", "ok": finite})
    # the content checks judge the frames that will be KEPT - held head frames and grid padding repeat on purpose
    off, n = accepted_slice(p)
    kept = images[off: off + n] if images.shape[0] >= off + n > off else images
    a0 = int(p["accepted"][0]) if kept is not images else 0
    means = kept[..., :3].float().mean(dim=(1, 2, 3))
    dark = [a0 + int(i) for i in torch.nonzero(means < min_mean).flatten().tolist()]
    checks.append({"name": "not black", "ok": len(dark) <= max(0, int(0.05 * len(means))), "frames": dark[:50]})
    if kept.shape[0] > 1:
        step = (kept[1:, ..., :3] - kept[:-1, ..., :3]).abs().mean(dim=(1, 2, 3))
        frozen = float((step < 1e-4).float().mean())
        checks.append({"name": "not frozen", "ok": frozen <= freeze_max, "value": round(frozen, 4)})
    ph = int(p.get("pad_head") or 0)
    if context is not None and int(p["context_frames"]) and context.shape[0] == int(p["context_frames"]) \
            and context.shape[1:3] == images.shape[1:3]:
        seam = float((images[ph: ph + context.shape[0], ..., :3] - context[..., :3]).abs().mean())
        checks.append({"name": "continues the accepted frames", "ok": seam <= seam_max, "value": round(seam, 4)})
    if diff_report and diff_max > 0:
        checks.append({"name": "difference from the source", "ok": float(diff_report.get("max", 0)) <= diff_max,
                       "value": diff_report.get("max")})
    base = (memory or {}).get("baseline")
    if base and colour_drift_max > 0 and (memory or {}).get("scene") == p.get("scene"):
        # the scene's look so far, from accepted segments only; a scene's first segment has no baseline (skipped)
        off, n = accepted_slice(p)
        acc = images[off: off + n]
        if acc.shape[0]:
            st = frame_stats(acc.float())
            de = math.sqrt(sum((a - b) ** 2 for a, b in zip(st["lab_mean"], base["lab_mean"])))
            checks.append({"name": "keeps the scene's colour", "ok": de <= colour_drift_max, "value": round(de, 3),
                           "max": colour_drift_max})
    checks += edit_checks(images, p, plate, mask, protect_px=protect_px, background_max=background_max,
                          mask_jump_max=mask_jump_max, strict_masks=strict_masks)
    if plate is not None and context is not None and seam_ratio_max > 0:
        checks += seam_check(images, p, plate, context, seam_ratio_max)
    verdict = "pass" if all(c["ok"] for c in checks) else "fail"
    return {"c2c_contract": QC, "c2c_contract_version": "1.0", "segment": seg["segment"], "attempt": seg["attempt"],
            "verdict": verdict, "checks": checks}


def seam_check(images: torch.Tensor, p: dict, plate: torch.Tensor, context: torch.Tensor, limit: float) -> list[dict]:
    """The step from the previous accepted frame into this segment, against the plate's own step at the same place
    (Rev 3 Part 5's source-relative seam ratio; ported from open-source-higgsfield's qc.py): real motion at a seam
    moves the plate too, a generation seam does not. Segments that start without continuation (a cut, the first) have
    no seam to judge."""
    from mmx_utils import body_double as BD
    off, n = accepted_slice(p)
    if not int(p.get("context_frames") or 0) or off < 1 or n < 2 or plate.shape[0] < off + 2             or context.shape[0] < 1 or tuple(context.shape[1:3]) != tuple(images.shape[1:3]):
        return []

    def lumas(frames):
        h, w = max(8, frames.shape[1] // 4), max(8, frames.shape[2] // 4)
        return [BD.resize2d(BD.luma(f), h, w) for f in frames]

    k = min(n, 13)
    out = lumas(torch.cat([context[-1:, ..., :3], images[off: off + k, ..., :3]]).float())
    src = lumas(plate[off - 1: off + k, ..., :3].float())
    seam, seam_src = BD.seam_ratio(out, 1), BD.seam_ratio(src, 1)
    return [{"name": "no seam at the segment start", "ok": seam <= max(limit, limit * seam_src),
             "value": round(seam, 3), "source": round(seam_src, 3), "max": limit}]


def edit_checks(images: torch.Tensor, p: dict, plate: torch.Tensor | None, mask: torch.Tensor | None, *,
                protect_px: int = 4, background_max: float = 0.008, mask_jump_max: float = 0.35,
                strict_masks: bool = False) -> list[dict]:
    """Rev 3 Part 13 / 14 for edits, on the ACCEPTED frames (global numbers in the report):
    background preservation - outside the mask (grown by protect_px) the result is the plate within background_max;
    mask stability - area jumps and dropouts between frames (a warning unless strict_masks)."""
    out: list[dict] = []
    off, n = accepted_slice(p)
    a0 = int(p["accepted"][0])
    m = None
    if mask is not None:
        m = mask if mask.ndim == 3 else mask[..., 0]
        if m.shape[0] != images.shape[0] or tuple(m.shape[1:3]) != tuple(images.shape[1:3]):
            raise ValueError(f"H3 QC Gate: the mask is {tuple(m.shape)} and the images {tuple(images.shape[:3])} - "
                             f"give the mask of THIS segment's window (Job Controller matte_window).")
        m = m[off: off + n].float()
    if plate is not None:
        if plate.shape[0] != images.shape[0] or tuple(plate.shape[1:3]) != tuple(images.shape[1:3]):
            raise ValueError(f"H3 QC Gate: the plate is {tuple(plate.shape[:3])} and the images "
                             f"{tuple(images.shape[:3])} - give the plate of THIS segment's window.")
        res, ref = images[off: off + n, ..., :3].float(), plate[off: off + n, ..., :3].float()
        if m is not None:
            grown = m[:, None]
            if protect_px > 0:
                k = 2 * int(protect_px) + 1
                grown = F.max_pool2d(grown, k, stride=1, padding=int(protect_px))
            outside = (grown[:, 0] < 0.5)
        else:
            outside = torch.ones(res.shape[:3], dtype=torch.bool)
        diff = (res - ref).abs().amax(-1)
        sel = diff[outside]
        worst = float(sel.max()) if sel.numel() else 0.0
        per = [float(diff[t][outside[t]].max()) if outside[t].any() else 0.0 for t in range(diff.shape[0])]
        bad = [a0 + t for t, v in enumerate(per) if v > background_max]
        out.append({"name": "keeps the plate outside the edit", "ok": worst <= background_max,
                    "value": round(worst, 5), "mean": round(float(sel.mean()) if sel.numel() else 0.0, 6),
                    "max": background_max, "frames": bad[:100]})
    if m is not None and m.shape[0]:
        area = m.flatten(1).mean(1)
        flagged, dropouts = [], []
        for t in range(1, area.shape[0]):
            prev, cur = float(area[t - 1]), float(area[t])
            if max(prev, cur) > 0 and abs(cur - prev) / max(prev, cur) > mask_jump_max:
                flagged.append(a0 + t)
            if cur == 0 and prev > 0 and t + 1 < area.shape[0] and float(area[t + 1]) > 0:
                dropouts.append(a0 + t)
        steady = not flagged and not dropouts
        out.append({"name": "mask is steady", "ok": steady or not strict_masks, "warning": not steady,
                    "jumps": flagged[:100], "dropouts": dropouts[:100],
                    "area_min": round(float(area.min()), 5), "area_max": round(float(area.max()), 5)})
    return out


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
                                   tooltip="0 = do not judge the difference report."),
                    io.String.Input("memory", default="", optional=True, force_input=True,
                                    tooltip="From H3 Subject Memory: judge the segment against its scene's look."),
                    io.Float.Input("colour_drift_max", default=12.0, min=0.0, max=100.0, step=0.1,
                                   tooltip="With a memory: the largest mean-colour distance (CIE76 delta E) from the "
                                           "scene's accepted frames. 0 = do not judge."),
                    io.Image.Input("plate", optional=True,
                                   tooltip="The plate of this segment's window (edit tasks): outside the mask the "
                                           "result must stay the plate."),
                    io.Mask.Input("mask", optional=True,
                                  tooltip="The edit mask of this segment's window (e.g. from matte_window)."),
                    io.Int.Input("protect_px", default=4, min=0, max=128,
                                 tooltip="The mask is grown by this many pixels before the plate check."),
                    io.Float.Input("background_max", default=0.008, min=0.0, max=1.0, step=0.001,
                                   tooltip="Largest difference from the plate allowed outside the edit (2/255)."),
                    io.Float.Input("mask_jump_max", default=0.35, min=0.0, max=1.0, step=0.01,
                                   tooltip="A frame whose mask area changes more than this share is flagged."),
                    io.Boolean.Input("strict_masks", default=False,
                                     tooltip="Fail the segment on mask jumps / dropouts (otherwise a warning)."),
                    io.Float.Input("seam_ratio_max", default=1.5, min=0.0, max=20.0, step=0.1,
                                   tooltip="With a plate and continuation: the step into this segment may be at most "
                                           "this many times the steps around it, and this many times the plate's own "
                                           "ratio there (real motion is not a seam). 0 = do not judge.")],
            outputs=[io.String.Output("qc_report"), io.Boolean.Output("passed"), io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, segment, images, context_images=None, diff_report="", min_mean=0.01, freeze_max=0.5,
                seam_max=0.08, diff_max=0.0, memory="", colour_drift_max=12.0, plate=None, mask=None, protect_px=4,
                background_max=0.008, mask_jump_max=0.35, strict_masks=False, seam_ratio_max=1.5) -> io.NodeOutput:
        seg = json.loads(segment)
        dr = json.loads(diff_report) if str(diff_report or "").strip() else None
        mem = _json(memory, MEMORY, "H3 QC Gate") if str(memory or "").strip() else None
        rep = qc_checks(images, seg, context_images, min_mean=min_mean, freeze_max=freeze_max, seam_max=seam_max,
                        diff_report=dr, diff_max=diff_max, memory=mem, colour_drift_max=colour_drift_max,
                        plate=plate, mask=mask, protect_px=protect_px, background_max=background_max,
                        mask_jump_max=mask_jump_max, strict_masks=strict_masks, seam_ratio_max=seam_ratio_max)
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


def _client_id() -> str | None:
    """The browser that queued the running prompt. Core sends progress and outputs only to a prompt's client_id, so
    the runs auto-continue queues carry it too - without it the person who clicked sees none of them."""
    try:
        from server import PromptServer  # type: ignore[import-not-found]
        return getattr(PromptServer.instance, "client_id", None)
    except Exception:  # noqa: BLE001 - outside a server (tests): no browser to report to
        return None


def post_prompt(prompt: dict, extra_pnginfo: dict | None) -> str:
    """Queue ``prompt`` through the server's own public POST /prompt (validation, ids, on_prompt hooks all apply).
    ``extra_pnginfo`` carries the workflow (for saved files) and the auto-continue token."""
    import ssl
    import urllib.request

    payload = {"prompt": prompt, "extra_data": {"extra_pnginfo": extra_pnginfo or {}}}
    cid = _client_id()
    if cid:
        payload["client_id"] = cid
    body = json.dumps(payload).encode("utf-8")
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
    for node in nxt.values():
        # core's IsChangedCache writes each node's fingerprint INTO the running prompt; a copy that kept them would
        # replay this run's values and the next run would come entirely from the cache (seen live, source-free job)
        if isinstance(node, dict):
            node.pop("is_changed", None)
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


def _pngs(d: str) -> list[str]:
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")] if os.path.isdir(d) else []


def assembly_frames(j: J.Job) -> list:
    """The movie's frames in order: a PNG path, or (previous tail, frame, weight of frame) where a segment's first
    ``overlap_head`` frames cross-fade with the previous segment's tail (blend, dissolve) - linear, never a pure copy
    of either end. A tail missing (an older job, a short generation) leaves the frames as they are."""
    items: list = []
    prev_tail: list[str] = []
    for p in j.m["plan"]:
        s = j.seg(p["id"])
        d = _attempt_dir(j, int(p["id"]), int(s["takes"].get(s.get("take") or "main", len(s["attempts"]))))
        files = _pngs(d)
        n = min(int(p.get("overlap_head") or 0), len(prev_tail), len(files))
        n = next((k for k in range(n) if os.path.basename(prev_tail[k]) != os.path.basename(files[k])), n)
        items += [(prev_tail[k], files[k], (k + 1) / (n + 1)) for k in range(n)] + files[n:]
        prev_tail = _pngs(os.path.join(d, "tail"))
    return items


def _assembly_frame(item, h: int, w: int) -> torch.Tensor:
    if isinstance(item, str):
        return _read_png16(item)[:h, :w]
    a, b, wb = item
    return (_read_png16(a)[:h, :w] * (1.0 - wb) + _read_png16(b)[:h, :w] * wb)


def assemble(j: J.Job, spec: dict) -> str:
    """The final movie from every segment's chosen take, in plan order, streamed (8 frames in memory)."""
    codec = spec.get("deliver_codec", "prores")
    ext = {"h264": ".mp4", "prores": ".mov", "ffv1": ".mkv"}[codec]
    out = os.path.join(j.root, f"{j.m['name']}{ext}")
    items = assembly_frames(j)
    if not items:
        raise ValueError("H3 Job Commit: nothing to assemble.")
    first = _read_png16(items[0] if isinstance(items[0], str) else items[0][1])
    h, w = int(first.shape[0]), int(first.shape[1])
    if codec != "ffv1":
        w, h = w - w % 2, h - h % 2
    part = out + ".partial" + ext
    fps = Fraction(spec["fps"])
    wave, sr = _plate_audio(spec)
    akw = {"audio_rate": sr, "audio_channels": int(wave.shape[0])} if wave is not None else {}
    with StreamWriter(part, fps, w, h, codec=codec, ten_bit=codec != "h264", **akw) as wr:
        for i in range(0, len(items), 8):
            wr.write(torch.stack([_assembly_frame(it, h, w) for it in items[i:i + 8]]))
        samples = wr.write_audio(wave) if wave is not None else 0
    os.replace(part, out)
    seconds = samples / sr if wave is not None else None
    off_frames = (seconds - len(items) / float(fps)) * float(fps) if seconds is not None else None
    j.m["assembly"] = {"frames": len(items), "fps": spec["fps"], "audio": wave is not None,
                       "audio_seconds": round(seconds, 6) if seconds is not None else None,
                       "audio_offset_frames": round(off_frames, 3) if off_frames is not None else None,
                       "av_sync_ok": off_frames is None or abs(off_frames) < 1.0}
    j.save()
    return out


def _plate_audio(spec: dict):
    """The plate's sound for the job's frames ([channels, samples] float32, rate), or (None, 0)."""
    a = spec.get("audio")
    if not a:
        return None, 0
    from mmx_vendor.c2c_core.video.audio import LazyAudio
    la = LazyAudio(path=a["path"], stream_index=int(a.get("stream_index", 0)), start_sec=float(a["start_sec"]),
                   duration_sec=float(a["duration_sec"]), reverse=bool(a.get("reverse", False)))
    wave = la["waveform"]
    return wave[0].float(), int(la["sample_rate"])


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
                    io.Combo.Input("decision", options=list(DECISIONS), default=DECISIONS[0]),
                    io.Latent.Input("latent", optional=True,
                                    tooltip="The sampler's latent: kept with the accepted frames and handed to the "
                                            "next segment as context_latent.")],
            outputs=[io.String.Output("status"), io.String.Output("job_json"), io.String.Output("final_path")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, segment, images, qc_report="", decision=DECISIONS[0], latent=None) -> io.NodeOutput:
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
            off, _n = accepted_slice(p)
            keep = images[off: off + (a1 - a0)]
            if keep.shape[0] != a1 - a0:
                raise ValueError(f"H3 Job Commit: segment {sid} must cover frames {a0}-{a1 - 1}, the images give "
                                 f"{keep.shape[0]} of {a1 - a0} (generate {p['model_frames']} frames).")
            for k in range(keep.shape[0]):
                _write_png16(os.path.join(d, f"frame_{a0 + k:07d}.png"), keep[k])
            # the frames generated past the accepted end (blend / dissolve tail): the assembly cross-fades the next
            # segment's first overlap_head frames with them
            tail = images[off + (a1 - a0): off + (a1 - a0) + int(p.get("overlap_tail") or 0)]
            if tail.shape[0]:
                os.makedirs(os.path.join(d, "tail"), exist_ok=True)
                for k in range(tail.shape[0]):
                    _write_png16(os.path.join(d, "tail", f"frame_{a1 + k:07d}.png"), tail[k])
            digest = hashlib.sha256("".join(J.sha256_file(os.path.join(d, f))
                                            for f in sorted(os.listdir(d)) if f.endswith(".png")).encode()).hexdigest()
            outputs = {"frames_dir": d, "frames": int(keep.shape[0]), "frames_sha256": digest,
                       "tail_frames": int(tail.shape[0])}
            if latent is not None:
                outputs["latent"] = save_latent(d, latent)
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


# ── H3 Shot Planner (L9.77) ────────────────────────────────────────────────────────────────────────────────────────

_TIMELINE_EXAMPLE = ("# seconds | prompt | transition into the shot (cut, continue, dissolve N frames)\n"
                     "4 | a lighthouse keeper climbs the spiral stairs at dusk, lantern in hand | cut\n"
                     "3 | she reaches the lamp room and looks out to sea | continue\n"
                     "4 | a storm rolls in over the dark water, waves breaking on the rocks | dissolve 12")


class MiniMaxH3_ShotPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ShotPlanner", display_name="H3 Shot Planner", category="MiniMax H3/Job",
            description="A prompt timeline -> the shots of a long job. One shot per line: seconds | prompt | "
                        "transition. cut (the default) starts a new scene and carries no context across; continue "
                        "keeps the take going with a new prompt; dissolve N cross-fades N frames. Feed H3 Task "
                        "Spec's shot_plan.",
            inputs=[io.String.Input("timeline", default=_TIMELINE_EXAMPLE, multiline=True,
                                    tooltip="seconds | prompt | transition, one shot per line (# = a note). JSON "
                                            "works too: [{\"seconds\": 4, \"prompt\": \"...\", \"transition\": "
                                            "\"cut\"}, ...]."),
                    io.Float.Input("fps", default=24.0, min=1.0, max=240.0,
                                   tooltip="The job's rate (H3 generates at 24): seconds become whole frames.")],
            outputs=[io.String.Output("shot_plan", tooltip="h3.shot_plan v1"), io.String.Output("table"),
                     io.Int.Output("shots"), io.Int.Output("frames")],
        )

    @classmethod
    def execute(cls, timeline=_TIMELINE_EXAMPLE, fps=24.0) -> io.NodeOutput:
        try:
            shots = J.parse_timeline(timeline, fps)
        except J.JobError as exc:
            raise ValueError(f"H3 Shot Planner: {exc}") from exc
        fps_ = Fraction(fps).limit_denominator(1001000)
        total = int(shots[-1]["global"][1])
        plan = {"c2c_contract": SHOTS, "c2c_contract_version": "1.0", "fps": f"{fps_.numerator}/{fps_.denominator}",
                "frames": total, "scenes": int(shots[-1]["scene"]) + 1, "shots": shots}
        rows = []
        for s in shots:
            tr = s["transition"] + (f" {s['dissolve']}" if s["dissolve"] else "")
            rows.append(f"shot {s['id'] + 1:>2}  scene {s['scene'] + 1:>2}  {tr:<12} frames {s['global'][0]}-"
                        f"{s['global'][1] - 1} ({s['frames']})  {s['prompt'][:70]}")
        return io.NodeOutput(json.dumps(plan), "\n".join(rows), len(shots), total)


# ── H3 Reference Bank (L9.77) ──────────────────────────────────────────────────────────────────────────────────────

def _q16(img: torch.Tensor) -> np.ndarray:
    return np.rint(np.clip(img[..., :3].float().cpu().numpy(), 0.0, 1.0) * 65535.0).astype(np.uint16)


def dhash64(img: torch.Tensor) -> int:
    """64-bit difference hash: grey 9x8, each pixel brighter than its left neighbour = 1. Near-duplicates (a re-save,
    a small crop or resize) stay within a few bits."""
    g = img[..., :3].float().mean(-1)[None, None]
    s = F.interpolate(g, size=(8, 9), mode="area")[0, 0]
    return sum(1 << i for i, b in enumerate((s[:, 1:] > s[:, :-1]).flatten().tolist()) if b)


def _roles(text: str, n: int) -> list[str]:
    rs = [r.strip().lower() for r in str(text or "").split(",") if r.strip()] or ["identity"]
    bad = [r for r in rs if r not in ROLES]
    if bad:
        raise ValueError(f"H3 Reference Bank: unknown role(s) {', '.join(bad)} - use: {', '.join(ROLES)}.")
    if len(rs) == 1:
        return rs * n
    if len(rs) != n:
        raise ValueError(f"H3 Reference Bank: {len(rs)} roles for {n} images - give one role per image, or one for "
                         f"all.")
    return rs


def add_references(bank: dict | None, images: torch.Tensor, roles: list[str], folder: str, *,
                   near_max: int = 6) -> dict:
    """Add images to a bank. Each entry: id, role, sha256 of the 16-bit pixels (an exact duplicate), a 64-bit dHash
    (Hamming distance <= ``near_max`` from an earlier entry = a near duplicate), size, the saved PNG."""
    entries = [dict(e) for e in (bank or {}).get("entries") or []]
    for i in range(int(images.shape[0])):
        q = _q16(images[i])
        sha = hashlib.sha256(q.tobytes() + str(q.shape).encode()).hexdigest()
        dh = dhash64(images[i])
        dup = next(({"id": e["id"], "kind": "exact", "distance": 0} for e in entries if e["sha256"] == sha), None)
        if dup is None and entries:
            dist, near = min(((bin(dh ^ int(e["dhash"], 16)).count("1"), e) for e in entries), key=lambda t: t[0])
            if dist <= int(near_max):
                dup = {"id": near["id"], "kind": "near", "distance": dist}
        path = os.path.join(folder, f"{sha[:16]}.png")
        if not os.path.isfile(path):
            _write_png16(path, images[i])
        entries.append({"id": f"r{len(entries)}", "role": roles[i], "sha256": sha, "dhash": f"{dh:016x}",
                        "width": int(images.shape[2]), "height": int(images.shape[1]), "path": path,
                        "duplicate_of": dup})
    fp = J.fingerprint([[e["sha256"], e["role"]] for e in entries])[:16]
    return {"c2c_contract": BANK, "c2c_contract_version": "1.0", "entries": entries, "fingerprint": fp}


def _load_ref(path: str, who: str) -> torch.Tensor:
    if not os.path.isfile(path):
        raise ValueError(f"{who}: the reference image {path} is missing - rebuild the bank (H3 Reference Bank).")
    return _read_png16(path)


def _letterbox(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """[H, W, 3] fitted inside w x h, the aspect kept, black bars."""
    ih, iw = int(img.shape[0]), int(img.shape[1])
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    x = img[..., :3].float().movedim(-1, 0)[None]
    r = F.interpolate(x, size=(nh, nw), mode="area") if s < 1 else \
        F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros((h, w, 3))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r[0].movedim(0, -1)
    return out


def _stack_fitted(images: list[torch.Tensor]) -> torch.Tensor:
    """One IMAGE batch at the first image's size (the others letterboxed into it)."""
    h, w = int(images[0].shape[0]), int(images[0].shape[1])
    return torch.stack([im[..., :3].float() if tuple(im.shape[:2]) == (h, w) else _letterbox(im, w, h)
                        for im in images])


def _unique_by_sha(entries: list[dict], roles: tuple[str, ...]) -> list[dict]:
    seen, out = set(), []
    for e in entries:
        if e["role"] in roles and e["sha256"] not in seen:
            seen.add(e["sha256"])
            out.append(e)
    return out


def bank_gallery(entries: list[dict], tile: int = 192) -> torch.Tensor:
    from .review_package import _label

    cols = max(1, min(4, len(entries)))
    rows = max(1, -(-len(entries) // cols))
    sheet = torch.zeros((rows * tile, cols * tile, 3))
    for i, e in enumerate(entries):
        dup = e.get("duplicate_of")
        mark = "" if not dup else (f" = {dup['id']}" if dup["kind"] == "exact" else f" ~ {dup['id']} ({dup['distance']})")
        img = _label(_letterbox(_load_ref(e["path"], "H3 Reference Bank"), tile, tile), f"{e['id']} {e['role']}{mark}")
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
    return sheet[None]


class MiniMaxH3_ReferenceBank(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ReferenceBank", display_name="H3 Reference Bank", category="MiniMax H3/Job",
            description="Keeps a job's reference images with their roles (identity, face, outfit, scene, lighting "
                        "...), marks exact and near duplicates, and saves each once. Chain banks to add more. Its "
                        "fingerprint feeds H3 Task Spec's references; identity feeds the model's reference inputs; "
                        "H3 Subject Memory reads the bank.",
            inputs=[io.Image.Input("images"),
                    io.String.Input("roles", default="identity",
                                    tooltip="One role for every image, or one per image, comma-separated: "
                                            + ", ".join(ROLES) + "."),
                    io.String.Input("name", default="refs", tooltip="The bank's folder: output/h3_refs/<name>."),
                    io.String.Input("bank", optional=True, force_input=True,
                                    tooltip="An existing bank to add these images to."),
                    io.Int.Input("near_max", default=6, min=0, max=32, advanced=True,
                                 tooltip="Near duplicate: dHash bits that may differ (of 64).")],
            outputs=[io.String.Output("bank", tooltip="h3.reference_bank v1"),
                     io.String.Output("fingerprint", tooltip="Changes when an image or a role changes."),
                     io.Image.Output("gallery", tooltip="Every entry with its id, role and duplicate mark."),
                     io.Image.Output("identity", tooltip="The identity / face / head images (duplicates once).")],
        )

    @classmethod
    def execute(cls, images, roles="identity", name="refs", bank=None, near_max=6) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        prior = _json(bank, BANK, "H3 Reference Bank") if str(bank or "").strip() else None
        folder = str(job_folder(f"h3_refs/{J.safe_name(name)}", create=True, who="H3 Reference Bank"))
        b = add_references(prior, images, _roles(roles, int(images.shape[0])), folder, near_max=near_max)
        ident = _unique_by_sha(b["entries"], IDENTITY_ROLES)
        identity = _stack_fitted([_load_ref(e["path"], "H3 Reference Bank") for e in ident]) if ident else \
            _blocker("H3 Reference Bank: no image has the role identity, face or head - give one of them that role "
                     "(roles), or leave 'identity' unconnected.")
        return io.NodeOutput(json.dumps(b), b["fingerprint"], bank_gallery(b["entries"]), identity)


def _blocker(message: str):
    from comfy_execution.graph_utils import ExecutionBlocker
    return ExecutionBlocker(message)


# ── H3 Subject Memory (L9.77) ──────────────────────────────────────────────────────────────────────────────────────

def _chosen_frames(j: J.Job, sid: int) -> list[str]:
    s = j.seg(sid)
    return _pngs(_attempt_dir(j, sid, int(s["takes"].get(s.get("take") or "main", len(s["attempts"])))))


def subject_memory(j: J.Job, bank: dict | None, seg_id: int | None, *, max_anchors: int = 4) -> dict:
    """Rebuilt from the job manifest on every read, so a rejected generation can never be in it: identity from the
    bank; approved examples (each accepted segment's last frame, nearest first) and the appearance baseline from the
    ACCEPTED segments of the CURRENT scene only - a cut starts over with the bank alone."""
    plan = {int(p["id"]): p for p in j.m["plan"]}
    if seg_id is None:
        nxt = j.next_segment()
        seg_id = int(nxt["id"]) if nxt is not None else (max(plan) if plan else None)
    scene = plan[seg_id].get("scene") if seg_id in plan else None
    scene_segs = sorted((sid for sid, p in plan.items()
                         if sid != seg_id and p.get("scene") == scene and j.seg(sid)["status"] == "accepted"),
                        key=lambda sid: (abs(sid - (seg_id or 0)), -sid))
    examples, files = [], []
    for sid in scene_segs:
        fr = _chosen_frames(j, sid)
        if not fr:
            continue
        files += fr
        if len(examples) < int(max_anchors):
            examples.append({"segment": sid, "take": j.seg(sid).get("take") or "main", "path": fr[-1]})
    baseline, sampled = None, 0
    if files:
        files.sort()
        pick = files[:: max(1, len(files) // 16)][:16]
        baseline = frame_stats(torch.stack([_read_png16(f) for f in pick]))
        sampled = len(pick)
    entries = (bank or {}).get("entries") or []
    identity = [{"id": e["id"], "role": e["role"], "path": e["path"], "sha256": e["sha256"]}
                for e in _unique_by_sha(entries, MEMORY_ROLES)]
    notes = []
    if bank is None:
        notes.append("no reference bank connected: identity comes only from approved frames")
    elif not _unique_by_sha(entries, IDENTITY_ROLES):
        notes.append("the bank has no identity / face / head image: identity comes only from approved frames")
    if seg_id is not None and not scene_segs:
        notes.append(f"scene {0 if scene is None else scene + 1} has no accepted segment yet: no approved examples, no "
                     f"colour baseline (nothing is carried across a cut)")
    anchors = [{k: a.get(k) for k in ("sha256", "source_sha256", "candidate", "path", "time", "note")}
               for a in j.m.get("anchors") or [] if a.get("status") == "approved"]        # never a rejected one
    mem = {"c2c_contract": MEMORY, "c2c_contract_version": "1.0", "job": j.m["name"], "segment": seg_id,
           "scene": scene, "identity": identity, "anchors": anchors, "examples": examples, "baseline": baseline,
           "baseline_frames": sampled, "notes": notes}
    J._atomic_json(os.path.join(j.root, "memory.json"), mem)
    return mem


class MiniMaxH3_SubjectMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SubjectMemory", display_name="H3 Subject Memory", category="MiniMax H3/Job",
            description="What the subject looks like so far, rebuilt from the job on every run: the bank's identity "
                        "images plus approved frames from ACCEPTED segments of the current scene (never a rejected "
                        "one; a cut starts over). anchors feed the model's reference inputs; memory feeds H3 QC "
                        "Gate's colour continuity check.",
            inputs=[io.String.Input("segment", optional=True, force_input=True,
                                    tooltip="H3 Job Controller's segment (this run's scene)."),
                    io.String.Input("job", default="h3_job", tooltip="Used when 'segment' is not connected."),
                    io.String.Input("bank", optional=True, force_input=True, tooltip="From H3 Reference Bank."),
                    io.Int.Input("max_anchors", default=4, min=0, max=32,
                                 tooltip="Approved frames to add after the identity images (nearest segments first).")],
            outputs=[io.Image.Output("anchors", tooltip="Identity images, then approved frames of this scene."),
                     io.String.Output("memory", tooltip="h3.subject_memory v1 (also written to <job>/memory.json)."),
                     io.String.Output("notes")],
        )

    @classmethod
    def _open(cls, segment, job):
        seg = json.loads(segment) if str(segment or "").strip() else {}
        j = J.Job.open(_jobs_dir(), seg.get("job") or job, create=False, recover=False)
        return j, (int(seg["segment"]) if "segment" in seg else None)

    @classmethod
    def fingerprint_inputs(cls, segment=None, job="h3_job", bank=None, max_anchors=4):
        try:
            j, sid = cls._open(segment, job)
            return f"{j.state_token()}|{sid}|{_sha(bank or '')}|{max_anchors}"
        except Exception:  # noqa: BLE001 - let execute say what is wrong
            return f"none|{_sha(segment or '')}"

    @classmethod
    def execute(cls, segment=None, job="h3_job", bank=None, max_anchors=4) -> io.NodeOutput:
        try:
            j, sid = cls._open(segment, job)
        except J.JobError as exc:
            raise ValueError(f"H3 Subject Memory: {exc}") from exc
        b = _json(bank, BANK, "H3 Subject Memory") if str(bank or "").strip() else None
        mem = subject_memory(j, b, sid, max_anchors=max_anchors)
        imgs = [_load_ref(e["path"], "H3 Subject Memory") for e in mem["identity"]] + \
               [_read_png16(e["path"]) for e in mem["examples"]]
        anchors = _stack_fitted(imgs) if imgs else _blocker(
            "H3 Subject Memory: nothing to anchor this scene yet - connect an H3 Reference Bank with an identity "
            "image, or leave 'anchors' unconnected for prompt-only shots.")
        notes = "; ".join(mem["notes"]) or (f"{len(mem['identity'])} identity image(s), {len(mem['examples'])} "
                                            f"approved frame(s) from scene {(mem['scene'] or 0) + 1}")
        return io.NodeOutput(anchors, json.dumps(mem), notes)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_TaskSpec": MiniMaxH3_TaskSpec,
    "MiniMaxH3_CompatibilityValidator": MiniMaxH3_CompatibilityValidator,
    "MiniMaxH3_SegmentPlanner": MiniMaxH3_SegmentPlanner,
    "MiniMaxH3_JobController": MiniMaxH3_JobController,
    "MiniMaxH3_QCGate": MiniMaxH3_QCGate,
    "MiniMaxH3_JobCommit": MiniMaxH3_JobCommit,
    "MiniMaxH3_ShotPlanner": MiniMaxH3_ShotPlanner,
    "MiniMaxH3_ReferenceBank": MiniMaxH3_ReferenceBank,
    "MiniMaxH3_SubjectMemory": MiniMaxH3_SubjectMemory,
}
