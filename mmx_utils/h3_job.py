"""H3 job engine: the state machine, the persistent manifest, segment records, retries, takes, rollback and resume
(owner A21 Revision 3 Part 8; ledger L9.75). Pure Python + file I/O - no ComfyUI import - so every rule is testable
on the CPU. Design: docs/evidence/A21/phase2/DESIGN_L9.75_job_engine.md in the workspace.

One job = one folder ``<root>/<name>/`` holding ``job.json`` (rewritten atomically after every change) and a
``job.lock`` while a run owns it. The H3 Continuation session (h3_session.py) is unchanged; a job can drive it as one
task kind.

Rules that matter (Rev 3 8.3 / 8.4):
* a REJECTED, FAILED, STALE or SUPERSEDED segment is never continuation context - the next segment waits for an
  ACCEPTED one;
* retries are bounded (``max_retries``); past it the job waits for review instead of looping;
* choosing another take of a segment, or a changed source / reference, marks only the segments that depended on it
  stale - unrelated accepted work is kept;
* a run that died mid-segment (crash, interrupt) leaves the segment "generating"; the next open marks it failed and the
  job continues from the last accepted segment.

Original code, GPL-3.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

SCHEMA = 1
CONTRACT = "h3.job"

STATES = ("CREATED", "VALIDATING", "PLANNED", "PREPARING_MEDIA", "DETECTING", "TRACKING", "BUILDING_REFERENCES",
          "WAITING_FOR_APPROVAL", "GENERATING", "SEGMENT_QC", "WAITING_FOR_REVIEW", "COMMITTING", "ASSEMBLING",
          "FINAL_QC", "PUBLISHING", "COMPLETED", "PAUSED", "CANCELLED", "FAILED")
_SETUP = ("CREATED", "VALIDATING", "PLANNED", "PREPARING_MEDIA", "DETECTING", "TRACKING", "BUILDING_REFERENCES",
          "WAITING_FOR_APPROVAL")
TERMINAL = ("COMPLETED", "CANCELLED")
STOPPED = ("PAUSED", "CANCELLED", "FAILED", "COMPLETED", "WAITING_FOR_REVIEW", "WAITING_FOR_APPROVAL")


def _allowed() -> dict[str, set[str]]:
    a: dict[str, set[str]] = {s: set() for s in STATES}
    for i, s in enumerate(_SETUP):                              # setup moves forward, skipping stages a task skips
        a[s] |= set(_SETUP[i + 1:]) | {"GENERATING"}
    a["GENERATING"] |= {"SEGMENT_QC", "COMMITTING", "WAITING_FOR_REVIEW"}
    a["SEGMENT_QC"] |= {"COMMITTING", "GENERATING", "WAITING_FOR_REVIEW"}
    a["WAITING_FOR_REVIEW"] |= {"COMMITTING", "GENERATING"}
    a["COMMITTING"] |= {"GENERATING", "ASSEMBLING", "WAITING_FOR_REVIEW"}
    a["ASSEMBLING"] |= {"FINAL_QC", "PUBLISHING", "COMPLETED"}
    a["FINAL_QC"] |= {"PUBLISHING", "COMPLETED", "WAITING_FOR_REVIEW"}
    a["PUBLISHING"] |= {"COMPLETED"}
    a["FAILED"] |= {"GENERATING", "PLANNED"}
    for s in ("ASSEMBLING", "FINAL_QC", "PUBLISHING", "COMPLETED"):
        a[s] |= {"GENERATING"}                                  # a rollback / another take / a change re-opens it
    for s in STATES:
        if s not in TERMINAL:
            a[s] |= {"PAUSED", "CANCELLED", "FAILED"}
    a["PAUSED"] |= set(_SETUP) | {"GENERATING", "SEGMENT_QC", "WAITING_FOR_REVIEW", "COMMITTING", "ASSEMBLING"}
    for s in STATES:
        a[s].discard(s)
    return a


ALLOWED = _allowed()
SEG_STATUSES = ("planned", "generating", "qc", "waiting review", "accepted", "rejected", "failed", "stale",
                "superseded")
_DEAD = ("rejected", "failed", "stale", "superseded")
LOCK_STALE_S = 6 * 3600


class JobError(ValueError):
    """Something about the job the user has to fix, said in plain English."""


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._")
    if not s:
        raise JobError("the job needs a name (letters, digits, '.', '_' or '-').")
    return s[:80]


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def sha256_file(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _atomic_json(path: str, data: dict) -> None:
    """Write ``.tmp`` then replace. On Windows a replace is refused while another process has the file open (a
    viewer, an editor, an antivirus scan): retry briefly instead of failing the run (measured live, L9.75)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


# ── the segment planner (Rev 3 7.3: six handle meanings, never mixed) ───────────────────────────────────────────────

def plan_segments(first: int, count: int, *, segment_frames: int, context: int = 0, blend: int = 0,
                  lookbehind: int = 0, lookahead: int = 0, source_first: int | None = None,
                  source_last: int | None = None, grid: tuple[int, int] | None = (17, 5),
                  lead_context: int = 0, tail: int = 0, hold_head: int = 0) -> list[dict]:
    """Segments covering global frames first .. first + count - 1.

    * ``segment_frames``: the model-local length of every segment (H3: 17k + 5; ``grid`` checks it, None = any);
    * ``context``: frames at the head of segment k > 0 that are the PREVIOUS ACCEPTED segment's output, given to the
      model as fixed continuation context (H3 Continuation's latent tail) - not generated again, not accepted again;
    * ``blend``: frames generated by both neighbours and cross-faded (segment overlap) - accepted once, from the later
      segment's side after the fade;
    * ``lookbehind`` / ``lookahead``: tracking handles - read from the source, never generated;
    * model-grid padding: the last segment's frames past the end (generated, then discarded) - ``pad_tail``;
    * the accepted ranges tile first .. first + count exactly;
    * ``lead_context``: segment 0 also continues from accepted frames before ``first`` (a shot that continues the
      previous take, L9.77); ``tail``: frames past the end the last segment keeps generating, as the overlap tail a
      following dissolve cross-fades with (taken from what would otherwise be grid padding);
    * ``hold_head``: a segment with no continuation context first generates this many copies of its first frame,
      then discards them (``pad_head``; the PRB workflow's AV handles, L9.76) - the model's first frames settle on a
      held picture instead of on the cut. It takes model frames from the segment, so ``segment_frames`` stays on the
      grid.
    """
    L, first, count = int(segment_frames), int(first), int(count)
    if count <= 0:
        raise JobError("the range to plan is empty.")
    if grid is not None and L % grid[0] != grid[1]:
        raise JobError(f"segment_frames {L} is not on the model grid ({grid[0]}k + {grid[1]}: e.g. "
                       f"{', '.join(str(grid[0] * k + grid[1]) for k in (1, 2, 4, 6))}).")
    context, blend = int(context), int(blend)
    if context < 0 or blend < 0 or context + blend >= L:
        raise JobError(f"context ({context}) + blend ({blend}) must be shorter than a segment ({L} frames).")
    hold_head = int(hold_head)
    if hold_head < 0 or hold_head + blend >= L - 1:
        raise JobError(f"hold_head ({hold_head}) + blend ({blend}) must leave room in a segment ({L} frames).")
    end = first + count
    src_lo = first if source_first is None else int(source_first)
    src_hi = end if source_last is None else int(source_last) + 1
    out: list[dict] = []
    start, k = first - int(lead_context), 0
    accepted_from = first
    while accepted_from < end:
        head = int(lead_context) if k == 0 else context
        hold = hold_head if head == 0 else 0                    # held copies of frame ``start`` before it
        gen_lo = start + head                                   # first frame the model generates
        gen_hi = start + L - hold                               # model-local end
        acc_lo = accepted_from
        acc_hi = min(end, gen_hi - (blend if gen_hi < end else 0))
        if acc_hi <= acc_lo:
            raise JobError(f"segment {k} would accept nothing - lower context / blend or raise segment_frames.")
        last = acc_hi >= end
        tail_kept = min(int(tail), max(0, gen_hi - end)) if last else 0
        seg = {
            "id": k, "model_frames": L,
            "global": [start, min(gen_hi, end + tail_kept)],     # timeline frames the segment touches
            "context_frames": head, "context_from": (k - 1) if k > 0 else None,
            "overlap_head": (blend if k > 0 else 0), "overlap_tail": (blend if gen_hi < end else tail_kept),
            "pad_head": hold, "pad_tail": max(0, gen_hi - end - tail_kept),
            "accepted": [acc_lo, acc_hi],
            "track_window": [max(src_lo, start - int(lookbehind)), min(src_hi, gen_hi + int(lookahead))],
            "generated": [gen_lo, min(gen_hi, end)],
        }
        out.append(seg)
        if acc_hi >= end:
            break
        accepted_from = acc_hi
        # the next model input starts with ``context`` accepted frames; it then generates from acc_hi, its first
        # ``blend`` frames being the ones this segment generated but did not accept (the cross-fade)
        start = acc_hi - context
        k += 1
    return out


TRANSITIONS = ("cut", "continue", "dissolve")


def parse_timeline(text: str, fps: float, first: int = 0) -> list[dict]:
    """A prompt timeline -> shots. One shot per line: ``seconds | prompt | transition`` (the transition INTO the shot:
    cut - the default -, continue, or ``dissolve N``); or JSON: a list of {seconds, prompt, transition}. Lines starting
    with # are notes."""
    raw = str(text or "").strip()
    items: list[dict] = []
    if raw.startswith("["):
        try:
            items = [dict(x) for x in json.loads(raw)]
        except (ValueError, TypeError) as exc:
            raise JobError(f"the timeline JSON is not a list of shots ({exc}).") from exc
    else:
        for n, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 2:
                raise JobError(f"timeline line {n}: write 'seconds | prompt | transition' ({line!r}).")
            items.append({"seconds": parts[0], "prompt": parts[1],
                          "transition": parts[2] if len(parts) > 2 else "cut", "line": n})
    if not items:
        raise JobError("the timeline has no shots.")
    shots, at, scene = [], int(first), 0
    for i, it in enumerate(items):
        where = f"timeline line {it.get('line', i + 1)}"
        try:
            secs = float(it["seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JobError(f"{where}: the first field must be the shot length in seconds.") from exc
        frames = int(round(secs * float(fps)))
        if frames <= 0:
            raise JobError(f"{where}: {secs} s is less than one frame.")
        tr = str(it.get("transition") or "cut").strip().lower()
        kind, _, arg = tr.partition(" ")
        if kind not in TRANSITIONS:
            raise JobError(f"{where}: transition {tr!r} - one of cut, continue, dissolve N.")
        dissolve = 0
        if kind == "dissolve":
            try:
                dissolve = int(arg or 0)
            except ValueError as exc:
                raise JobError(f"{where}: write 'dissolve 12' (frames).") from exc
            if dissolve <= 0 or dissolve >= frames:
                raise JobError(f"{where}: a dissolve of {dissolve} frames must be shorter than the shot ({frames}).")
        if i == 0:
            kind, dissolve = "cut", 0                       # the first shot starts the film
        if kind != "continue" and i:
            scene += 1
        shots.append({"id": i, "scene": scene, "seconds": secs, "frames": frames, "global": [at, at + frames],
                      "prompt": str(it.get("prompt", "")), "transition": kind, "dissolve": dissolve})
        at += frames
    return shots


def plan_shots(shots: list[dict], *, segment_frames: int, context: int = 0, blend: int = 0, lookbehind: int = 0,
               lookahead: int = 0, grid: tuple[int, int] | None = (17, 5), hold_head: int = 0) -> list[dict]:
    """Segments for a shot list. A CUT starts its shot with no continuation context (nothing leaks across a cut); a
    CONTINUE shot's first segment continues from the previous shot's last segment; a DISSOLVE N shot is a cut whose
    first N frames are cross-faded with N frames the previous shot keeps generating past its end."""
    out: list[dict] = []
    for i, shot in enumerate(shots):
        a, b = shot["global"]
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        tail = int(nxt["dissolve"]) if nxt is not None and nxt["transition"] == "dissolve" else 0
        lead = int(context) if (shot["transition"] == "continue" and out) else 0
        segs = plan_segments(a, b - a, segment_frames=segment_frames, context=context, blend=blend,
                             lookbehind=lookbehind, lookahead=lookahead, grid=grid, lead_context=lead, tail=tail,
                             hold_head=hold_head)
        if tail and segs[-1]["overlap_tail"] < tail:
            raise JobError(f"shot {i + 1}: the dissolve into shot {i + 2} needs {tail} frames past its end and its "
                           f"last segment has {segs[-1]['overlap_tail']} to spare - shorten the dissolve, or change "
                           f"the shot length or segment_frames.")
        base = len(out)
        for sg in segs:
            sg["id"] += base
            if sg["context_from"] is not None:
                sg["context_from"] += base
            sg.update(shot=shot["id"], scene=shot["scene"], prompt=shot["prompt"], transition="within")
        head = segs[0]
        head["transition"] = shot["transition"]
        if lead:
            head["context_from"] = out[-1]["id"]
        if shot["transition"] == "dissolve" and out:
            head["overlap_head"] = int(shot["dissolve"])
        out.extend(segs)
    return out


_KEY_FIELDS = ("accepted", "global", "context_from", "context_frames", "overlap_head", "overlap_tail", "shot", "prompt",
               "transition", "pad_head")


def plan_key(seg: dict) -> str:
    """What makes a planned segment the same segment: a changed key re-opens it (and what continues from it)."""
    return json.dumps([seg.get(k) for k in _KEY_FIELDS])


def check_tiling(segments: list[dict], first: int, count: int) -> list[str]:
    problems, at = [], int(first)
    for s in segments:
        a, b = s["accepted"]
        if a != at:
            problems.append(f"segment {s['id']}: accepted starts at {a}, expected {at}")
        at = b
    if at != int(first) + int(count):
        problems.append(f"the accepted ranges end at {at}, not at {int(first) + int(count)}")
    return problems


# ── the job ────────────────────────────────────────────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, root: str):
        self.root = root
        self.path = os.path.join(root, "job.json")
        self.m: dict[str, Any] = {}

    # -- open / create ------------------------------------------------------------------------------------------------
    @classmethod
    def open(cls, base_dir: str, name: str, *, spec: dict | None = None, create: bool = True,
             recover: bool = True) -> "Job":
        """``recover``: mark a segment left mid-run by a crash as failed - only where a run STARTS (the controller);
        nodes later in the same run open with recover=False, or they would fail the segment being made."""
        root = os.path.join(base_dir, safe_name(name))
        j = cls(root)
        if os.path.isfile(j.path):
            try:
                with open(j.path, encoding="utf-8") as f:
                    j.m = json.load(f)
            except (OSError, ValueError) as exc:
                raise JobError(f"the job file {j.path} is unreadable ({exc}) - restore it or start a new job "
                               "under another name.") from exc
            if j.m.get("c2c_contract") != CONTRACT or int(j.m.get("schema", 0)) != SCHEMA:
                raise JobError(f"{j.path} is not an H3 job of this version (schema {j.m.get('schema')}).")
            if recover:
                j._recover_interrupted()
            if spec is not None:
                j.update_spec(spec)
            return j
        if not create:
            raise JobError(f"there is no job '{name}' in {base_dir}.")
        os.makedirs(root, exist_ok=True)
        now = time.time()
        j.m = {"c2c_contract": CONTRACT, "c2c_contract_version": "1.0", "schema": SCHEMA, "job_id": fingerprint(
            [root, now]), "name": safe_name(name), "created": now, "state": "CREATED",
            "history": [{"state": "CREATED", "time": now, "reason": "new job"}], "spec": spec or {},
            "fingerprints": dict((spec or {}).get("fingerprints") or {}), "plan": [], "segments": {},
            "retry_history": [], "outputs": {}, "versions": _versions(), "auto_runs": 0}
        j.save()
        return j

    def save(self) -> None:
        self.m["updated"] = time.time()
        _atomic_json(self.path, self.m)

    # -- state machine ------------------------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self.m["state"]

    def transition(self, new: str, reason: str = "") -> None:
        if new not in STATES:
            raise JobError(f"unknown job state {new!r}")
        cur = self.state
        if new == cur:
            return
        if new not in ALLOWED[cur]:
            raise JobError(f"the job is {cur}; it cannot go to {new} (next: {', '.join(sorted(ALLOWED[cur]))}).")
        if new == "PAUSED":
            self.m["paused_from"] = cur
        self.m["state"] = new
        self.m["history"].append({"state": new, "time": time.time(), "reason": reason})
        self.save()

    def pause(self, reason: str = "paused by the user") -> None:
        if self.state not in TERMINAL and self.state != "PAUSED":
            self.transition("PAUSED", reason)

    def resume(self, reason: str = "resumed by the user") -> None:
        if self.state == "PAUSED":
            back = self.m.get("paused_from") or "GENERATING"
            self.transition(back if back in ALLOWED["PAUSED"] else "GENERATING", reason)
        elif self.state == "FAILED":
            self.transition("GENERATING", reason)

    def cancel(self, reason: str = "cancelled by the user") -> None:
        if self.state not in TERMINAL:
            self.transition("CANCELLED", reason)

    # -- plan ---------------------------------------------------------------------------------------------------------
    def set_plan(self, segments: list[dict]) -> None:
        """Store the planner's segments. A segment already accepted is kept when its plan entry is the SAME
        (``plan_key``: ranges, context, overlaps, shot prompt) and so is the segment it continues from."""
        old = self.m["segments"]
        old_keys = {str(p["id"]): plan_key(p) for p in self.m.get("plan") or []}
        self.m["plan"] = [dict(s) for s in segments]
        fresh = {}
        for s in segments:
            key = str(s["id"])
            prev = old.get(key)
            cf = s.get("context_from")
            chain_kept = cf is None or fresh.get(str(cf), {}).get("status") == "accepted"
            if prev and prev.get("range") == s["accepted"] and prev["status"] == "accepted" \
                    and old_keys.get(key) == plan_key(s) and chain_kept:
                fresh[key] = prev
            else:
                fresh[key] = {"id": s["id"], "status": "planned", "range": list(s["accepted"]), "attempts": [],
                              "takes": {}, "take": None, "context_from": s.get("context_from")}
        self.m["segments"] = fresh
        if self.state in _SETUP:
            self.transition("PLANNED", f"{len(segments)} segments planned")
        else:
            self.save()

    def seg(self, seg_id: int) -> dict:
        s = self.m["segments"].get(str(int(seg_id)))
        if s is None:
            raise JobError(f"segment {seg_id} is not in this job's plan.")
        return s

    def plan_of(self, seg_id: int) -> dict:
        for p in self.m["plan"]:
            if int(p["id"]) == int(seg_id):
                return p
        raise JobError(f"segment {seg_id} is not in this job's plan.")

    def next_segment(self) -> dict | None:
        """The first segment (plan order) that is not accepted; None when every segment is accepted."""
        for p in self.m["plan"]:
            s = self.seg(p["id"])
            if s["status"] != "accepted":
                return s
        return None

    def context_ok(self, seg_id: int) -> tuple[bool, str]:
        cf = self.seg(seg_id).get("context_from")
        if cf is None:
            return True, ""
        c = self.seg(cf)
        if c["status"] != "accepted":
            return False, f"segment {seg_id} continues from segment {cf}, which is {c['status']} (not accepted)"
        return True, ""

    # -- one attempt --------------------------------------------------------------------------------------------------
    def seed_for(self, seg_id: int, policy: str = "new seed") -> int:
        base = int((self.m.get("spec") or {}).get("seed", 0))
        n = len(self.seg(seg_id)["attempts"])
        if policy == "same seed" and n:
            return int(self.seg(seg_id)["attempts"][-1]["seed"])
        return base + int(seg_id) + 1000 * n

    def begin(self, seg_id: int, *, seed: int, inputs: dict | None = None, changes: str = "",
              alternate: bool = False) -> dict:
        """Start an attempt. ``alternate``: an extra take of an ACCEPTED segment - recorded beside the chosen take,
        which stays accepted and keeps feeding the segments after it (8.4 "branch alternate take")."""
        ok, why = self.context_ok(seg_id)
        if not ok:
            raise JobError(why + " - accept or retry it first.")
        s = self.seg(seg_id)
        if s["status"] == "accepted" and not alternate:
            raise JobError(f"segment {seg_id} is already accepted - roll back, choose another take, or generate an "
                           f"alternate take.")
        if alternate and s["status"] != "accepted":
            raise JobError(f"segment {seg_id} has no accepted take yet - an alternate needs one to sit beside.")
        if self.state != "GENERATING" and not alternate:
            self.transition("GENERATING", f"segment {seg_id}")
        attempt = {"n": len(s["attempts"]) + 1, "seed": int(seed), "inputs": inputs or {}, "changes": changes,
                   "started": time.time(), "verdict": None, "outputs": {}, "alternate": bool(alternate)}
        s["attempts"].append(attempt)
        if not alternate:
            s["status"] = "generating"
        self.m["current"] = int(seg_id)
        self.save()
        return attempt

    def record_qc(self, seg_id: int, report: dict) -> None:
        s = self.seg(seg_id)
        s["attempts"][-1]["qc"] = report
        if s["attempts"][-1].get("alternate"):              # an extra take: the accepted segment and the job stay put
            self.save()
            return
        s["status"] = "qc"
        if self.state == "GENERATING":
            self.transition("SEGMENT_QC", f"segment {seg_id} QC: {report.get('verdict')}")
        else:
            self.save()

    def accept(self, seg_id: int, outputs: dict | None = None, *, take: str = "main", note: str = "") -> None:
        s = self.seg(seg_id)
        if not s["attempts"]:
            raise JobError(f"segment {seg_id} has no attempt to accept.")
        att = s["attempts"][-1]
        att["outputs"] = _with_hashes(outputs or {})
        if att.get("alternate"):                                # kept as a named take; the chosen take is unchanged
            att["verdict"] = "kept"
            s["takes"][take if take != "main" else f"take{att['n']}"] = att["n"]
            self.save()
            return
        att["verdict"] = "accepted"
        s["takes"][take] = att["n"]
        s["take"] = take
        s["status"] = "accepted"
        s["note"] = note
        self._commit_state(f"segment {seg_id} accepted ({take})")

    def reject(self, seg_id: int, reason: str, *, max_retries: int = 3) -> str:
        """Returns 'retry' or 'review' (retries used up)."""
        s = self.seg(seg_id)
        if s["attempts"]:
            s["attempts"][-1]["verdict"] = "rejected"
            s["attempts"][-1]["reason"] = reason
        s["status"] = "rejected"
        self.m["retry_history"].append({"segment": int(seg_id), "time": time.time(), "reason": reason})
        if len(s["attempts"]) >= int(max_retries) + 1:
            s["status"] = "waiting review"
            self._to("WAITING_FOR_REVIEW", f"segment {seg_id}: {len(s['attempts'])} attempts rejected - {reason}")
            return "review"
        self._to("GENERATING", f"segment {seg_id} rejected, retrying: {reason}")
        return "retry"

    def fail(self, seg_id: int, reason: str) -> None:
        s = self.seg(seg_id)
        if s["attempts"]:
            s["attempts"][-1]["verdict"] = "failed"
            s["attempts"][-1]["reason"] = reason
        s["status"] = "failed"
        self.save()

    def hold_for_review(self, seg_id: int, reason: str) -> None:
        s = self.seg(seg_id)
        s["status"] = "waiting review"
        s["note"] = reason
        self._to("WAITING_FOR_REVIEW", f"segment {seg_id}: {reason}")

    # -- takes / rollback / upstream changes (8.4) --------------------------------------------------------------------
    def branch(self, seg_id: int, take: str, *, attempt: int | None = None) -> None:
        """Keep an attempt as a named alternate take (its outputs stay recorded)."""
        s = self.seg(seg_id)
        n = int(attempt) if attempt else (s["attempts"][-1]["n"] if s["attempts"] else 0)
        if not n or n > len(s["attempts"]):
            raise JobError(f"segment {seg_id} has no attempt {attempt}.")
        s["takes"][str(take)] = n
        self.save()

    def choose_take(self, seg_id: int, take: str) -> list[int]:
        """Make ``take`` the accepted output of the segment; segments continuing from it become stale. Returns them."""
        s = self.seg(seg_id)
        if take not in s["takes"]:
            raise JobError(f"segment {seg_id} has no take {take!r} (takes: {', '.join(s['takes']) or 'none'}).")
        if s.get("take") == take and s["status"] == "accepted":
            return []
        s["take"], s["status"] = take, "accepted"
        stale = self._stale_after(int(seg_id), f"segment {seg_id} now uses take {take}")
        if self.state in ("COMPLETED", "ASSEMBLING", "FINAL_QC", "PUBLISHING") and stale:
            self.transition("GENERATING", f"take {take} of segment {seg_id} re-opens {len(stale)} segment(s)")
        self.save()
        return stale

    def rollback(self, to_segment: int | None) -> list[int]:
        """Keep segments up to ``to_segment`` (None = keep none); every later one is superseded - generated again with a
        new seed; its files stay until a purge."""
        order = [int(p["id"]) for p in self.m["plan"]]
        if to_segment is not None and int(to_segment) not in order:
            raise JobError(f"segment {to_segment} is not in the plan.")
        cut = -1 if to_segment is None else order.index(int(to_segment))
        gone = []
        for sid in order[cut + 1:]:
            s = self.seg(sid)
            if s["status"] != "planned":
                s["status"] = "superseded"
                gone.append(sid)
            s["take"] = None
        if self.state in ("COMPLETED", "ASSEMBLING", "FINAL_QC", "PUBLISHING", "FAILED"):
            self.transition("GENERATING", f"rolled back to segment {to_segment}")
        self.save()
        return gone

    def update_spec(self, spec: dict) -> list[int]:
        """A new task spec. Changed source / reference fingerprints make ONLY the segments that used them stale."""
        old_fp = dict(self.m.get("fingerprints") or {})
        new_fp = dict(spec.get("fingerprints") or {})
        changed = {k for k in set(old_fp) | set(new_fp) if old_fp.get(k) != new_fp.get(k)}
        self.m["spec"] = spec
        self.m["fingerprints"] = new_fp
        stale: list[int] = []
        if changed:
            for p in self.m["plan"]:
                s = self.seg(p["id"])
                if s["status"] != "accepted" or not s["attempts"]:
                    continue
                used = set((s["attempts"][s["takes"].get(s["take"], len(s["attempts"])) - 1].get("inputs") or {})
                           .get("uses", []))
                if not used or used & changed:
                    s["status"] = "stale"
                    stale.append(int(p["id"]))
            if stale and self.state in ("COMPLETED", "ASSEMBLING", "FINAL_QC", "PUBLISHING"):
                self.transition("GENERATING", f"{', '.join(sorted(changed))} changed: {len(stale)} segment(s) stale")
        self.save()
        return stale

    def _stale_after(self, seg_id: int, why: str) -> list[int]:
        """Segments whose continuation chain leads back to ``seg_id`` (accepted ones become stale)."""
        order = [int(p["id"]) for p in self.m["plan"]]
        dependent = {int(seg_id)}
        out = []
        for sid in order:
            s = self.seg(sid)
            cf = s.get("context_from")
            if cf is not None and int(cf) in dependent:
                dependent.add(sid)
                if s["status"] in ("accepted", "qc", "waiting review"):
                    s["status"] = "stale"
                    s["note"] = why
                    out.append(sid)
        return out

    # -- commit / completion ------------------------------------------------------------------------------------------
    def _commit_state(self, reason: str) -> None:
        if self.state in ("GENERATING", "SEGMENT_QC", "WAITING_FOR_REVIEW"):
            self.transition("COMMITTING", reason)
        if self.next_segment() is None:
            self.transition("ASSEMBLING", "every segment accepted")
        else:
            self.transition("GENERATING", reason + "; next segment")

    def _to(self, state: str, reason: str) -> None:
        if self.state == state:
            self.save()
        else:
            self.transition(state, reason)

    def complete(self, outputs: dict | None = None) -> None:
        if outputs:
            self.m["outputs"].update(_with_hashes(outputs))
        if self.state == "ASSEMBLING":
            self.transition("COMPLETED", "assembled")
        else:
            self.save()

    # -- crash recovery and locking -----------------------------------------------------------------------------------
    def _recover_interrupted(self) -> None:
        changed = False
        for s in self.m["segments"].values():
            if s["status"] in ("generating", "qc"):
                s["status"] = "failed"
                if s["attempts"]:
                    s["attempts"][-1]["verdict"] = "failed"
                    s["attempts"][-1]["reason"] = "interrupted (crash or cancel) - generating it again"
                s["note"] = "interrupted - generating it again"
                changed = True
        if changed:
            self.m["history"].append({"state": self.state, "time": time.time(),
                                      "reason": "an interrupted segment was marked failed on open"})
            self.save()

    def lock(self, owner: str) -> None:
        lp = os.path.join(self.root, "job.lock")
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"owner": owner, "pid": os.getpid(), "time": time.time()}, f)
            return
        except FileExistsError:
            pass
        try:
            with open(lp, encoding="utf-8") as f:
                held = json.load(f)
        except (OSError, ValueError):
            held = {"time": 0}
        if time.time() - float(held.get("time", 0)) > LOCK_STALE_S or held.get("owner") == owner:
            os.replace(lp, lp + ".stale")
            return self.lock(owner)
        raise JobError(f"the job '{self.m.get('name')}' is in use by another run ({held.get('owner')}); "
                       f"wait for it, or delete {lp} if that run is gone.")

    def unlock(self) -> None:
        try:
            os.remove(os.path.join(self.root, "job.lock"))
        except FileNotFoundError:
            pass

    # -- summaries ----------------------------------------------------------------------------------------------------
    def summary(self) -> dict:
        segs = [self.seg(p["id"]) for p in self.m["plan"]]
        by: dict[str, int] = {}
        for s in segs:
            by[s["status"]] = by.get(s["status"], 0) + 1
        nxt = self.next_segment()
        return {"job": self.m["name"], "state": self.state, "segments": len(segs), "by_status": by,
                "next": None if nxt is None else nxt["id"], "auto_runs": self.m.get("auto_runs", 0),
                "path": self.path}

    def state_token(self) -> str:
        """Changes whenever the job does (IS_CHANGED for the controller)."""
        return fingerprint([self.state, self.m.get("updated"), [(k, v["status"], len(v["attempts"]))
                                                                for k, v in sorted(self.m["segments"].items())]])


def _with_hashes(outputs: dict) -> dict:
    out = {}
    for k, v in (outputs or {}).items():
        if isinstance(v, str) and os.path.isfile(v):
            out[k] = {"path": v, "sha256": sha256_file(v), "bytes": os.path.getsize(v)}
        else:
            out[k] = v
    return out


def _versions() -> dict:
    v: dict[str, Any] = {"engine": "h3_job/1"}
    for mod in ("torch", "av", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    try:
        import comfyui_version  # type: ignore[import-not-found]
        v["comfyui"] = comfyui_version.__version__
    except Exception:  # noqa: BLE001
        v["comfyui"] = None
    return v
