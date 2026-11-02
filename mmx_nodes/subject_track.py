"""H3 Multi-Subject Selector, H3 Subject Re-Identification and H3 Temporal Subject Tracker (owner A21 Rev 3 Part 10;
ledger L9.78 slice 3).

The detector and the tracker are ComfyUI core's own SAM 3 / 3.1 (core 0.39: SAM3_Detect, SAM3_VideoTrack,
SAM3_TrackToMask), called at run time - no SAM code lives here. Selection, loss detection and re-identification are in
mmx_utils/subject_track.py (ported from open-source-higgsfield, Apache-2.0). The rule Part 10 adds: a choice the
numbers do not support goes to REVIEW - the subject is never changed silently.

The tracker is a resumable job: one window per queue run, its state and masks in ComfyUI's output folder
(h3_jobs/<job>/track/), the next run queued automatically. Original nodes, GPL-3.0. Design: docs/evidence/A21/phase2/
DESIGN_L9.78_character_swap.md in the workspace.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import body_double as BD  # noqa: E402
from mmx_utils import h3_job as J  # noqa: E402
from mmx_utils import subject_track as ST  # noqa: E402
from mmx_vendor.c2c_core.video import reader as _reader  # noqa: E402

from . import h3_job_nodes as HJ  # noqa: E402
from .media_window import C2C_VIDEO, is_handle  # noqa: E402
from .review_package import _label  # noqa: E402
from .segment_io import job_folder  # noqa: E402

CATEGORY = "MiniMax H3/Subject"
TRACK_CONTRACT = "h3.subject_track"
TRACK_VERSION = "1.0"
BACKENDS = ("SAM 3 (core)", "masks input (precomputed)")
REVIEW = ("wait", "continue with candidate", "end the track here", "track again")
CONFIRM = ("auto when confident", "always ask", "pick index")


# ── shared helpers ───────────────────────────────────────────────────────────────────────────────────────────────

def _bbox_out(box) -> list[list[dict]]:
    """One frame's box as core's BOUNDING_BOX value (a list of box dicts per frame)."""
    if box is None:
        return [[]]
    return [[{"x": float(box[0]), "y": float(box[1]), "width": float(box[2] - box[0]),
              "height": float(box[3] - box[1])}]]


def _split_masks(masks, per_frame: list[list[dict]], frame: int) -> list | None:
    """core SAM3_Detect's individual masks are ONE batch for all frames; give back the masks of ``frame``."""
    if masks is None:
        return None
    m = masks if masks.ndim == 3 else masks.unsqueeze(0)
    counts = [len(f) for f in per_frame]
    if int(m.shape[0]) == sum(counts) and frame < len(counts):
        a = sum(counts[:frame])
        return [m[i] for i in range(a, a + counts[frame])]
    if len(per_frame) <= 1 and counts and int(m.shape[0]) == counts[0]:
        return [m[i] for i in range(int(m.shape[0]))]
    return None                                             # a union mask: the boxes stand alone


def _frame_boxes(per_frame: list[list[dict]], frame: int) -> list[dict]:
    if not per_frame:
        return []
    return per_frame[frame] if frame < len(per_frame) else (per_frame[0] if len(per_frame) == 1 else [])


def gallery(frame: np.ndarray, cands: list[dict], ranked: list[dict], chosen: int | None, tile: int = 160) -> torch.Tensor:
    """The candidates side by side, best first: each crop with its mask tinted, '#index score', the pick outlined."""
    by = {c["index"]: c for c in cands}
    seen = [r["index"] for r in ranked if r["index"] in by]
    order = seen + [c["index"] for c in cands if c["index"] not in seen]
    score = {r["index"]: r["score"] for r in ranked}
    h, w = int(frame.shape[0]), int(frame.shape[1])
    tiles = []
    for k in order:
        c = by[k]
        x0, y0, x1, y1 = c["box"]
        pad = 0.1 * max(x1 - x0, y1 - y0)
        ya, yb = int(max(0, y0 - pad)), int(min(h, y1 + pad))
        xa, xb = int(max(0, x0 - pad)), int(min(w, x1 + pad))
        crop = np.array(frame[ya:yb, xa:xb, :3], dtype=np.float32)
        if crop.size == 0:
            crop = np.zeros((8, 8, 3), np.float32)
        else:
            mm = c["mask"][ya:yb, xa:xb] > 0.5
            crop[mm] = crop[mm] * 0.7 + np.array([0.1, 0.9, 0.3], np.float32) * 0.3
        t = HJ._letterbox(torch.from_numpy(crop), tile, tile)
        lab = f"#{k} " + (f"{score[k]:.2f}" if k in score else "too far") + (" PICK" if k == chosen else "")
        t = _label(t, lab)
        if k == chosen:
            t[:3], t[-3:], t[:, :3], t[:, -3:] = (torch.tensor([0.2, 0.9, 0.3]),) * 4
        tiles.append(t)
    if not tiles:
        tiles = [_label(torch.zeros((tile, tile, 3)), "no candidate")]
    return torch.cat(tiles, dim=1)[None]


def _review(*names_and_msg):
    msg = names_and_msg[-1]
    return [HJ._blocker(msg) for _ in names_and_msg[:-1]]


# ── H3 Multi-Subject Selector ────────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_MultiSubjectSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MultiSubjectSelector", display_name="H3 Multi-Subject Selector", category=CATEGORY,
            description="Picks the person to replace from core SAM 3 Detect's candidates and gives them a stable "
                        "subject id. Auto ranks by size, by similarity to a reference and by a position preference; "
                        "when two candidates are too close to call it stops and shows them (gallery) instead of "
                        "guessing - set selection = index. A subject lock keeps the person of an earlier run.",
            inputs=[io.Image.Input("images", tooltip="The frame(s) SAM 3 Detect ran on."),
                    io.BoundingBox.Input("bboxes", force_input=True,
                                         tooltip="Candidates: core SAM 3 Detect's bboxes, from the text "
                                                 "'person:20' (':20' = up to 20 people; plain 'person' finds one)."),
                    io.Mask.Input("masks", optional=True,
                                  tooltip="SAM 3 Detect's masks with individual_masks on: one per candidate."),
                    io.Image.Input("reference_images", optional=True,
                                   tooltip="The performer to find: their appearance (colours) is compared."),
                    io.Mask.Input("reference_masks", optional=True,
                                  tooltip="Where the performer is in the reference images (default: the whole image)."),
                    io.String.Input("subject_lock", optional=True, force_input=True,
                                    tooltip="An H3 subject from an earlier run: that person is kept, with their id."),
                    io.Int.Input("frame", default=0, min=0, max=100000, tooltip="Which of the images to pick on."),
                    io.Combo.Input("selection", options=["auto", "index"], default="auto"),
                    io.Int.Input("index", default=0, min=0, max=256, tooltip="With selection = index: the '#' number "
                                 "in the gallery."),
                    io.Combo.Input("position_preference", options=list(ST.POSITIONS), default="none"),
                    io.Float.Input("size_weight", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                    io.Float.Input("reference_weight", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True,
                                   tooltip="Weight of the appearance match to the reference images."),
                    io.Float.Input("position_weight", default=0.5, min=0.0, max=10.0, step=0.05, advanced=True),
                    io.Float.Input("ambiguity_margin", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="Auto stops for review when the best two scores are closer than this."),
                    io.Float.Input("lock_min_score", default=0.5, min=-1.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="With a subject lock: below this match score it stops for review.")],
            outputs=[io.String.Output("subject_id"), io.Mask.Output("mask"), io.BoundingBox.Output("bbox"),
                     io.String.Output("subject", tooltip="H3 subject (id, box, appearance): for the tracker and "
                                                         "re-identification."),
                     io.Image.Output("gallery", tooltip="The candidates, best first; the pick outlined."),
                     io.String.Output("report")],
        )

    @classmethod
    def execute(cls, images, bboxes, masks=None, reference_images=None, reference_masks=None, subject_lock=None,
                frame=0, selection="auto", index=0, position_preference="none", size_weight=1.0,
                reference_weight=1.0, position_weight=0.5, ambiguity_margin=0.05,
                lock_min_score=0.5) -> io.NodeOutput:
        who = "H3 Multi-Subject Selector"
        imgs = images if images.ndim == 4 else images.unsqueeze(0)
        f = min(int(frame), int(imgs.shape[0]) - 1)
        img = imgs[f, ..., :3].float().cpu().numpy()
        h, w = int(img.shape[0]), int(img.shape[1])
        per_frame = ST.frames_of(bboxes)
        cands = ST.candidates(img, _frame_boxes(per_frame, int(frame)), _split_masks(masks, per_frame, int(frame)))
        if not cands:
            raise ValueError(f"{who}: no candidate on frame {int(frame)} - SAM 3 Detect found nothing above its "
                             "threshold there. Lower its threshold or pick another frame.")
        lock = ST.parse_subject(subject_lock, who)
        ref = None
        if reference_images is not None:
            r = reference_images if reference_images.ndim == 4 else reference_images.unsqueeze(0)
            ref = ST.appearance([r[i, ..., :3] for i in range(int(r.shape[0]))],
                                None if reference_masks is None else reference_masks if reference_masks.ndim == 3
                                else reference_masks.unsqueeze(0))
        if lock is not None:
            last = ST.scaled_box(lock["box"], lock["frame_size"], (w, h))
            ranked = ST.rank_reid(w, cands, np.asarray(lock["hist"]), last)
            verdict, why = ST.decide(ranked, min_score=float(lock_min_score), margin=float(ambiguity_margin))
            sid, how = lock["subject_id"], "subject lock"
        elif selection == "index":
            ranked = ST.rank_selection(cands, w, h, size_weight=size_weight, ref_hist=ref,
                                       reference_weight=reference_weight, position=position_preference,
                                       position_weight=position_weight)
            if int(index) not in {c["index"] for c in cands}:
                raise ValueError(f"{who}: there is no candidate #{int(index)} - the candidates are "
                                 f"{sorted(c['index'] for c in cands)} (see the gallery).")
            ranked.sort(key=lambda r: r["index"] != int(index))
            verdict, why, sid, how = "accept", f"#{int(index)} chosen by the user", None, "index"
        else:
            ranked = ST.rank_selection(cands, w, h, size_weight=size_weight, ref_hist=ref,
                                       reference_weight=reference_weight, position=position_preference,
                                       position_weight=position_weight)
            verdict, why = ST.decide(ranked, margin=float(ambiguity_margin))
            sid, how = None, "auto"
        pick = ranked[0]["index"] if verdict == "accept" and ranked else None
        rep = {"frame": int(frame), "candidates": len(cands), "selection": how, "decision": verdict, "why": why,
               "ranking": ranked}
        sheet = gallery(img, cands, ranked, pick)
        if verdict != "accept":
            msg = (f"{who}: {why} - the subject is not picked. Look at the gallery, then set selection = index and "
                   f"index = the '#' number (or connect a reference of the performer).")
            rep["subject_id"] = None
            return io.NodeOutput(*_review("id", "mask", "bbox", "subject", msg), sheet, json.dumps(rep))
        c = next(x for x in cands if x["index"] == pick)
        sid = sid or ST.subject_id(int(frame), c["box"], w, h)
        rep["subject_id"] = sid
        subject = ST.subject_json(sid, int(frame), c["box"], w, h, c["hist"], c["area"], selection=how,
                                  score=ranked[0]["score"])
        return io.NodeOutput(sid, torch.from_numpy((c["mask"] > 0.5).astype(np.float32))[None], _bbox_out(c["box"]),
                             subject, sheet, json.dumps(rep))


# ── H3 Subject Re-Identification ─────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_SubjectReIdentification(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SubjectReIdentification", display_name="H3 Subject Re-Identification",
            category=CATEGORY,
            description="Finds the tracked person again among new detections (after a cut, an occlusion, a lost "
                        "track). Score = appearance x colour similarity to the subject's memory - motion x how far "
                        "they moved (a fraction of the width). Below min_confidence, or with a second candidate "
                        "within margin, it stops for review and shows the candidates: the subject id is never "
                        "changed silently. Appearance is the body and clothing colours; no face model is used.",
            inputs=[io.Image.Input("images", tooltip="The frame(s) with the new detections."),
                    io.BoundingBox.Input("bboxes", force_input=True, tooltip="core SAM 3 Detect's bboxes."),
                    io.Mask.Input("masks", optional=True, tooltip="SAM 3 Detect's individual masks."),
                    io.String.Input("subject", force_input=True,
                                    tooltip="The subject's memory: H3 Multi-Subject Selector's or H3 Temporal Subject "
                                            "Tracker's subject."),
                    io.Int.Input("frame", default=0, min=0, max=100000),
                    io.Combo.Input("confirmation", options=list(CONFIRM), default=CONFIRM[0],
                                   tooltip="always ask: stop and show the candidates; pick index: take '#index'."),
                    io.Int.Input("index", default=0, min=0, max=256),
                    io.Float.Input("min_confidence", default=0.5, min=-1.0, max=1.0, step=0.01),
                    io.Float.Input("margin", default=0.05, min=0.0, max=1.0, step=0.01,
                                   tooltip="Review when the second candidate scores within this of the best."),
                    io.Float.Input("appearance_weight", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                    io.Float.Input("motion_weight", default=0.5, min=0.0, max=10.0, step=0.05, advanced=True,
                                   tooltip="Penalty per frame width of distance from where the subject was."),
                    io.Float.Input("search_distance", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True,
                                   tooltip="Ignore candidates farther than this (a fraction of the width).")],
            outputs=[io.String.Output("subject_id"), io.Mask.Output("mask"), io.BoundingBox.Output("bbox"),
                     io.Float.Output("confidence"), io.String.Output("subject"),
                     io.String.Output("ranking"), io.Image.Output("gallery")],
        )

    @classmethod
    def execute(cls, images, bboxes, subject, masks=None, frame=0, confirmation=CONFIRM[0], index=0,
                min_confidence=0.5, margin=0.05, appearance_weight=1.0, motion_weight=0.5,
                search_distance=1.0) -> io.NodeOutput:
        who = "H3 Subject Re-Identification"
        mem = ST.parse_subject(subject, who)
        if mem is None:
            raise ValueError(f"{who}: connect 'subject' (H3 Multi-Subject Selector or H3 Temporal Subject Tracker).")
        imgs = images if images.ndim == 4 else images.unsqueeze(0)
        f = min(int(frame), int(imgs.shape[0]) - 1)
        img = imgs[f, ..., :3].float().cpu().numpy()
        h, w = int(img.shape[0]), int(img.shape[1])
        per_frame = ST.frames_of(bboxes)
        cands = ST.candidates(img, _frame_boxes(per_frame, int(frame)), _split_masks(masks, per_frame, int(frame)))
        last = ST.scaled_box(mem["box"], mem["frame_size"], (w, h))
        ranked = ST.rank_reid(w, cands, np.asarray(mem["hist"]), last, appearance_weight=appearance_weight,
                              motion_weight=motion_weight, search_distance=search_distance)
        if confirmation == "pick index":
            if int(index) not in {r["index"] for r in ranked}:
                raise ValueError(f"{who}: there is no candidate #{int(index)} in range - the candidates are "
                                 f"{[r['index'] for r in ranked]} (see the gallery).")
            ranked.sort(key=lambda r: r["index"] != int(index))
            verdict, why = "accept", f"#{int(index)} confirmed by the user"
        elif confirmation == "always ask":
            verdict, why = "review", "confirmation = always ask"
        else:
            verdict, why = ST.decide(ranked, min_score=float(min_confidence), margin=float(margin))
        pick = ranked[0]["index"] if verdict == "accept" else None
        rank_json = json.dumps({"subject_id": mem["subject_id"], "frame": int(frame), "decision": verdict, "why": why,
                                "ranking": ranked})
        sheet = gallery(img, cands, ranked, pick)
        if verdict != "accept":
            msg = (f"{who}: {why} - subject {mem['subject_id']} is not re-identified. Look at the gallery; to "
                   "confirm a candidate set confirmation = pick index and index = its '#' number.")
            out = _review("id", "mask", "bbox", "confidence", "subject", msg)
            return io.NodeOutput(*out, rank_json, sheet)
        c = next(x for x in cands if x["index"] == pick)
        conf = float(np.clip(ranked[0]["score"], 0.0, 1.0))
        upd = dict(mem, frame=int(frame), box=[round(float(v), 2) for v in c["box"]], frame_size=[w, h],
                   area=round(float(c["area"]), 1), reidentified={"score": ranked[0]["score"], "why": why})
        return io.NodeOutput(mem["subject_id"], torch.from_numpy((c["mask"] > 0.5).astype(np.float32))[None],
                             _bbox_out(c["box"]), conf, json.dumps(upd), rank_json, sheet)


# ── H3 Temporal Subject Tracker (the track job) ──────────────────────────────────────────────────────────────────

def _core_track(model, frames: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """core SAM3_VideoTrack from ``seed`` (the subject on the FIRST frame) through ``frames`` -> [N, H, W] masks."""
    from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack
    td = SAM3_VideoTrack.execute(frames, model, initial_mask=(seed > 0.5).float()[None], max_objects=1).args[0]
    return SAM3_TrackToMask.execute(td, "0").args[0].float().cpu()


def _core_detect(model, conditioning, frame: torch.Tensor, threshold: float):
    """core SAM3_Detect on one frame [H, W, 3] -> (box dicts, masks)."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    out = SAM3_Detect.execute(model, frame[None], conditioning=conditioning, threshold=float(threshold),
                              individual_masks=True).args
    m, boxes = out[0], out[1]
    return (boxes[0] if boxes else []), [m[i].float().cpu() for i in range(int(m.shape[0]))]


_TRACK = _core_track                                         # tests replace these two
_DETECT = _core_detect
_POSTER = None                                               # tests replace it; None = H3 Job's post_prompt


class _Source:
    """The clip, read a window at a time at the proxy size: a lazy media handle (H3 Media Probe) or an IMAGE batch."""

    def __init__(self, media, images, proxy_width: int, who: str):
        self.media, self.images = media, images
        if media is not None:
            if not is_handle(media):
                raise ValueError(f"{who}: 'media' is not a C2C video handle - connect H3 Media Probe.")
            self.n = len(media.selected_indices())
            first = _reader.read(media, [0], fmt="float32")
            self.h, self.w = int(first.shape[1]), int(first.shape[2])
        elif images is not None:
            self.images = images if images.ndim == 4 else images.unsqueeze(0)
            self.n, self.h, self.w = (int(v) for v in self.images.shape[:3])
        else:
            raise ValueError(f"{who}: connect the clip - 'media' (H3 Media Probe) or 'images'.")
        if self.n < 1:
            raise ValueError(f"{who}: the clip has no frames.")
        pw = self.w if int(proxy_width) <= 0 or int(proxy_width) >= self.w else int(proxy_width)
        self.pw, self.ph = pw, max(2, int(round(self.h * pw / self.w)))

    def identity(self) -> str:
        if self.media is not None:
            src = getattr(self.media, "source", None)
            sel = [int(i) for i in self.media.selected_indices()]
            return J.fingerprint([str(getattr(src, "path", None) or getattr(src, "pattern", None) or src),
                                  getattr(src, "stat_key", None) and str(src.stat_key), sel[:1], sel[-1:], len(sel),
                                  list(getattr(self.media, "crop", None) or [])])
        im = self.images
        probe = torch.cat([im[0, ::8, ::8, :3].flatten(), im[-1, ::8, ::8, :3].flatten()]).float().cpu().numpy()
        return J.fingerprint([list(im.shape), J.fingerprint(probe.round(3).tolist())])

    def read(self, lo: int, hi: int) -> torch.Tensor:
        out = []
        for a in range(lo, hi, 16):
            b = min(hi, a + 16)
            if self.media is not None:
                arr = _reader.read(self.media, list(range(a, b)), fmt="float32")[..., :3]
                t = torch.from_numpy(np.ascontiguousarray(arr))
            else:
                t = self.images[a:b, ..., :3].float().cpu()
            if (int(t.shape[1]), int(t.shape[2])) != (self.ph, self.pw):
                t = F.interpolate(t.movedim(-1, 1), size=(self.ph, self.pw), mode="area").movedim(1, -1)
            out.append(t.contiguous())
        return torch.cat(out)


def _png(path: str, m) -> None:
    from PIL import Image
    Image.fromarray(((BD._np(m) > 0.5) * 255).astype(np.uint8)).save(path)


def _load(path: str) -> np.ndarray:
    from PIL import Image
    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) > 127).astype(np.float32)


def _resize_mask(m, h: int, w: int) -> np.ndarray:
    a = BD._np(m)
    if a.ndim == 3:
        a = a[0]
    return a if a.shape == (h, w) else (BD.resize2d(a, h, w) > 0.5).astype(np.float32)


def _track_root(job: str) -> Path:
    return job_folder(f"h3_jobs/{J.safe_name(job)}/track", create=True, who="H3 Temporal Subject Tracker")


def track_token(job: str) -> str:
    """Changes whenever any track of the job changes (the next queue must run)."""
    try:
        root = _track_root(job)
        stamps = sorted((p.parent.name, p.stat().st_mtime_ns) for p in root.glob("*/track.json"))
        return J.fingerprint(stamps) if stamps else "new"
    except Exception:  # noqa: BLE001 - let execute say what is wrong
        return "new"


class Track:
    """One track's state in <output>/h3_jobs/<job>/track/<key>/ (track.json + masks/ + seeds/ + review/)."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "track.json"
        self.s: dict = {}

    @classmethod
    def open(cls, base: Path, key: str, fresh: dict) -> "Track":
        t = cls(base / key)
        if t.path.is_file():
            t.s = json.loads(t.path.read_text(encoding="utf-8"))
            if t.s.get("c2c_contract") != TRACK_CONTRACT:
                raise ValueError(f"H3 Temporal Subject Tracker: {t.path} is not a subject track.")
            return t
        for d in ("masks", "seeds", "review"):
            (t.root / d).mkdir(parents=True, exist_ok=True)
        t.s = dict(fresh, c2c_contract=TRACK_CONTRACT, c2c_contract_version=TRACK_VERSION, created=time.time())
        t.save()
        return t

    def save(self) -> None:
        self.s["updated"] = time.time()
        J._atomic_json(str(self.path), self.s)

    def mask_path(self, i: int) -> str:
        return str(self.root / "masks" / f"{i:06d}.png")

    def event(self, frame: int, kind: str, **kw) -> None:
        self.s["events"].append(dict(frame=int(frame), kind=kind, time=round(time.time(), 2), **kw))

    @property
    def mean_hist(self):
        n = int(self.s["hist_n"])
        return np.asarray(self.s["hist_sum"]) / n if n else None

    def accept_frame(self, i: int, m: np.ndarray, frame: np.ndarray, conf: float, src_size, keep_hist: bool) -> None:
        _png(self.mask_path(i), m)
        b = BD.bbox(m)
        self.s["boxes"][str(i)] = [round(v, 2) for v in ST.scaled_box(b, self.s["proxy"], src_size)]
        self.s["confidence"][str(i)] = round(float(conf), 4)
        self.s["areas"] = (self.s["areas"] + [float((m > 0.5).sum())])[-5:]
        self.s["last_box"] = list(b)
        if keep_hist and int(self.s["hist_n"]) < 30:
            self.s["hist_sum"] = (np.asarray(self.s["hist_sum"]) + BD.color_hist(frame, m)).tolist()
            self.s["hist_n"] = int(self.s["hist_n"]) + 1

    def absent(self, i: int) -> None:
        _png(self.mask_path(i), np.zeros((self.s["proxy"][1], self.s["proxy"][0]), np.float32))
        self.s["boxes"][str(i)] = None
        self.s["confidence"][str(i)] = 0.0

    def reseed(self, i: int, m: np.ndarray) -> None:
        p = str(self.root / "seeds" / f"{i:06d}.png")
        _png(p, m)
        self.s["cursor"], self.s["seed"] = int(i), os.path.relpath(p, self.root)
        self.s["areas"], self.s["last_box"] = [float((m > 0.5).sum())], list(BD.bbox(m) or [0, 0, 0, 0])


def run_window(t: Track, src: _Source, *, track_fn, detect_fn, area_ratio: float, jump: float,
               validate_backward: bool, confidence_threshold: float, reid_interval: int, drift_similarity: float,
               max_gap: int, min_confidence: float, margin: float, motion_weight: float,
               search_distance: float) -> tuple[torch.Tensor, int]:
    """Track one window from the cursor; on a loss re-identify (accept, or stop for review). Returns the window's
    masks and its first frame."""
    s = t.s
    n, (pw, ph), src_size = int(s["n"]), s["proxy"], (src.w, src.h)
    i = int(s["cursor"])
    e = min(i + int(s["window"]), n)
    frames = src.read(i, e)
    seed = torch.from_numpy(_load(str(t.root / s["seed"])))
    wm = track_fn(frames, seed)
    if int(wm.shape[0]) != e - i:
        raise ValueError(f"H3 Temporal Subject Tracker: the tracker gave {int(wm.shape[0])} masks for {e - i} frames.")
    conf = [1.0] * (e - i)
    if validate_backward and e - i > 1:
        back = track_fn(frames.flip(0), wm[-1]).flip(0)
        conf = [ST.iou(wm[j], back[j]) for j in range(e - i)]
    lost, reason = None, None
    for j in range(e - i):
        m = _resize_mask(wm[j], ph, pw)
        fr = frames[j].numpy()
        reason = ST.loss_reason(m, s["areas"], s["last_box"], pw, area_ratio=area_ratio, jump=jump)
        if reason is None and validate_backward and conf[j] < float(confidence_threshold):
            reason = f"forward/backward agreement {conf[j]:.2f} is under {float(confidence_threshold):.2f}"
        if reason is None and int(reid_interval) > 0 and (i + j) % int(reid_interval) == 0 and t.mean_hist is not None:
            sim = BD.hist_similarity(BD.color_hist(fr, m), t.mean_hist)
            if sim < float(drift_similarity):
                reason = f"appearance similarity {sim:.2f} is under {float(drift_similarity):.2f} (another person?)"
        if reason is not None:
            lost = i + j
            break
        t.accept_frame(i + j, m, fr, conf[j], src_size, keep_hist=True)
    if lost is None:
        if e >= n:
            s["status"] = "done"
            t.event(n - 1, "complete", frames=n)
        else:
            t.reseed(e - 1, _resize_mask(wm[e - 1 - i], ph, pw))      # windows overlap by one frame (the seed)
        t.save()
        return wm, i
    t.event(lost, "lost", reason=reason)
    if detect_fn is None:
        s["status"], s["review"] = "waiting review", {"frame": lost, "ranking": [], "why":
                                                      f"lost at frame {lost} ({reason}); re-detection needs 'model' "
                                                      "and 'conditioning' (the text 'person') - connect them, or end "
                                                      "the track here"}
        t.event(lost, "review", why=s["review"]["why"])
        t.save()
        return wm, i
    f = lost
    while True:                       # up to max_gap frames may pass with no acceptable match (occluded, off frame)
        fr = src.read(f, f + 1)[0]
        boxes, masks = detect_fn(fr)
        cands = ST.candidates(fr.numpy(), boxes, masks)
        ranked = ST.rank_reid(pw, cands, t.mean_hist, s["last_box"], motion_weight=motion_weight,
                              search_distance=search_distance)
        verdict, why = ST.decide(ranked, min_score=float(min_confidence), margin=float(margin))
        if verdict == "accept" or f >= lost + int(max_gap) or f + 1 >= n:
            break
        t.absent(f)
        t.event(f, "absent", why=why)
        f += 1
    if lost == i and f == i and verdict == "accept":          # the reference stops here: the seed itself failed
        verdict, why = "review", f"lost at its own seed frame {i}; best: {why}"
    if verdict == "accept":
        c = next(x for x in cands if x["index"] == ranked[0]["index"])
        t.reseed(f, c["mask"])
        t.event(f, "reacquired", score=ranked[0]["score"], why=why)
    else:
        for c in cands:
            _png(str(t.root / "review" / f"f{f:06d}_c{c['index']}.png"), c["mask"])
        s["status"] = "waiting review"
        s["review"] = {"frame": f, "ranking": ranked, "why": f"lost at frame {lost} ({reason}); re-identification: "
                                                             f"{why}"}
        t.event(f, "review", why=s["review"]["why"], ranking=ranked[:5])
    t.save()
    return wm, i


def apply_review(t: Track, src: _Source, review: str, pick: int) -> str:
    s = t.s
    r = s.get("review") or {}
    f = int(r.get("frame", s["cursor"]))
    if review == "track again":                              # e.g. after connecting model + conditioning
        s["status"], s["review"] = "tracking", None
        t.event(f, "tracking again (user)")
        t.save()
        return f"tracking again from frame {int(s['cursor'])}"
    if review == "end the track here":
        for k in range(f, int(s["n"])):
            t.absent(k)
        s["status"], s["review"] = "done", None
        t.event(f, "ended by the user", frames_absent=int(s["n"]) - f)
        t.save()
        return f"the track ends at frame {f}: frames {f}..{int(s['n']) - 1} have no subject"
    p = t.root / "review" / f"f{f:06d}_c{int(pick)}.png"
    if not p.is_file():
        have = sorted(int(x.stem.split("_c")[1]) for x in (t.root / "review").glob(f"f{f:06d}_c*.png"))
        raise ValueError(f"H3 Temporal Subject Tracker: there is no candidate #{int(pick)} at frame {f} - the "
                         f"candidates are {have} (see the preview).")
    t.reseed(f, _load(str(p)))
    s["status"], s["review"] = "tracking", None
    t.event(f, "confirmed by the user", candidate=int(pick))
    t.save()
    return f"candidate #{int(pick)} confirmed at frame {f}; tracking continues"


def matte_handle(t: Track, src: _Source):
    """The finished track as a plate-size matte: masks/ upscaled once into matte/ (white = the subject), opened as a
    lazy image-sequence handle - H3 Job Controller's 'matte' reads only each segment's window of it."""
    from mmx_vendor.c2c_core.video import media as _media
    s = t.s
    d = t.root / "matte"
    stamp = d / "complete.json"
    want = {"frames": int(s["n"]), "size": [src.w, src.h], "updated": s.get("updated")}
    if not stamp.is_file() or json.loads(stamp.read_text(encoding="utf-8")) != want:
        d.mkdir(exist_ok=True)
        for i in range(int(s["n"])):
            _png(str(d / f"{i:06d}.png"), _resize_mask(_load(t.mask_path(i)), src.h, src.w))
        J._atomic_json(str(stamp), want)
    fps = float(getattr(src.media, "fps", 0) or 0) if src.media is not None else 0.0
    h, _facts = _media.probe_media(str(d / "%06d.png"), mode="image sequence", sequence_fps=fps or 24.0)
    return h


def preview(frames: torch.Tensor, masks: torch.Tensor, path: list) -> torch.Tensor:
    """The window's frames: the mask tinted, its box, and the subject's path so far (box centres)."""
    out = frames[..., :3].clone().float()
    h, w = int(out.shape[1]), int(out.shape[2])
    for j in range(int(out.shape[0])):
        m = _resize_mask(masks[j], h, w) > 0.5
        f = out[j]
        f[torch.from_numpy(m)] = f[torch.from_numpy(m)] * 0.6 + torch.tensor([0.1, 0.9, 0.3]) * 0.4
        b = BD.bbox(m)
        if b is not None:
            x0, y0, x1, y1 = (int(v) for v in b)
            f[y0:y0 + 2, x0:x1] = f[max(0, y1 - 2):y1, x0:x1] = torch.tensor([1.0, 0.85, 0.1])
            f[y0:y1, x0:x0 + 2] = f[y0:y1, max(0, x1 - 2):x1] = torch.tensor([1.0, 0.85, 0.1])
        for cx, cy in path:
            x, y = int(cx * w), int(cy * h)
            f[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = torch.tensor([1.0, 0.3, 0.3])
    return out


class MiniMaxH3_TemporalSubjectTracker(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TemporalSubjectTracker", display_name="H3 Temporal Subject Tracker", category=CATEGORY,
            description="Tracks one person through a clip of any length with core SAM 3, a window per run, and "
                        "queues the next run itself. A frame where the mask vanishes, shrinks under area_ratio of "
                        "its recent size or jumps sideways is a loss: the person is found again by appearance (H3 "
                        "Subject Re-Identification's rule) and tracking goes on - or, when the match is not clear, "
                        "the job STOPS for review and shows the candidates (the subject is never changed silently). "
                        "State and masks are kept in ComfyUI's output folder (h3_jobs/<job>/track/), so a crash or a "
                        "restart resumes where it stopped. Changing a setting starts a new track; the old one is kept.",
            inputs=[io.Model.Input("model", optional=True, tooltip="The SAM 3 / 3.1 checkpoint's MODEL (core)."),
                    C2C_VIDEO.Input("media", optional=True, tooltip="The clip as H3 Media Probe's lazy handle "
                                                                     "(any length: read a window at a time)."),
                    io.Image.Input("images", optional=True, tooltip="Or the clip as images (short clips)."),
                    io.Mask.Input("initial_mask", tooltip="The subject on the clip's first frame: H3 Multi-Subject "
                                                          "Selector's mask."),
                    io.String.Input("subject", force_input=True, tooltip="H3 Multi-Subject Selector's subject."),
                    io.Conditioning.Input("conditioning", optional=True,
                                          tooltip="The text 'person:20' (CLIP Text Encode with the SAM 3 checkpoint's CLIP; ':20' lets SAM 3 return up to 20 people - plain 'person' returns only one): needed to find the subject again after a loss."),
                    io.Mask.Input("masks", optional=True, tooltip="backend = masks input: a finished track of the "
                                                                  "whole clip (any tracker); losses are still checked."),
                    io.String.Input("job", default="track01", tooltip="Job name: the folder in "
                                                                      "output/h3_jobs/<job>/track/."),
                    io.Combo.Input("backend", options=list(BACKENDS), default=BACKENDS[0]),
                    io.Int.Input("window", default=100, min=2, max=2000, tooltip="Frames tracked per run."),
                    io.Combo.Input("review", options=list(REVIEW), default=REVIEW[0],
                                   tooltip="When the job waits for review: go on with candidate '#review_pick', end "
                                           "the track there (the subject has left), or track again (after you "
                                           "connected model + conditioning or changed nothing but want a retry)."),
                    io.Int.Input("review_pick", default=0, min=0, max=256),
                    io.Boolean.Input("auto_continue", default=True, tooltip="Queue the next window automatically."),
                    io.Combo.Input("action", options=["run", "restart"], default="run",
                                   tooltip="restart: start this track again on this run (the old files are kept "
                                           "aside); set it back to run afterwards."),
                    io.Int.Input("proxy_width", default=960, min=0, max=8192, advanced=True,
                                 tooltip="Track at this width (the reference uses 960); 0 = the source width."),
                    io.Float.Input("area_ratio", default=0.35, min=0.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="Loss: the mask shrinks under this share of the median of its last 5."),
                    io.Float.Input("jump", default=0.15, min=0.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="Loss: the centre moves more than this share of the width in one frame."),
                    io.Boolean.Input("validate_backward", default=False, advanced=True,
                                     tooltip="Also track each window backwards; frames where the two disagree "
                                             "(IoU under confidence_threshold) count as a loss. Twice the time."),
                    io.Float.Input("confidence_threshold", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Int.Input("reid_interval", default=0, min=0, max=1000, advanced=True,
                                 tooltip="Every N frames compare the tracked person's colours with the memory "
                                         "(0 = off): catches a tracker that moved to someone else."),
                    io.Float.Input("drift_similarity", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Int.Input("max_gap", default=0, min=0, max=1000, advanced=True,
                                 tooltip="Frames the subject may be gone (occluded, off frame: no candidate matches "
                                         "well enough) before the job asks. Those frames get an empty mask."),
                    io.Float.Input("min_confidence", default=0.5, min=-1.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="Re-identification below this score stops for review."),
                    io.Float.Input("margin", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Float.Input("motion_weight", default=0.5, min=0.0, max=10.0, step=0.05, advanced=True),
                    io.Float.Input("search_distance", default=1.0, min=0.0, max=2.0, step=0.05, advanced=True),
                    io.Float.Input("detection_threshold", default=0.4, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Int.Input("max_auto_runs", default=200, min=1, max=100000, advanced=True)],
            outputs=[io.Mask.Output("masks", tooltip="This run's window, at the proxy size."),
                     io.BoundingBox.Output("bboxes", tooltip="Every tracked frame so far, in source pixels "
                                                             "(an empty list where the subject is absent)."),
                     io.Float.Output("confidence", tooltip="The lowest per-frame confidence so far."),
                     io.String.Output("events"), io.String.Output("subject", tooltip="The subject's memory, updated."),
                     io.String.Output("status"), io.Image.Output("preview"),
                     io.String.Output("track_json", tooltip="Contract h3.subject_track: the folder, frames, boxes."),
                     C2C_VIDEO.Output("matte", tooltip="When the track is done: the subject's matte at the plate's "
                                                       "size, as a lazy handle (H3 Job Controller's matte).")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, job="track01", review=REVIEW[0], review_pick=0, action="run", **_kw):
        return f"{track_token(job)}|{review}|{review_pick}|{action}"

    @classmethod
    def execute(cls, initial_mask, subject, model=None, media=None, images=None, conditioning=None, masks=None,
                job="track01", backend=BACKENDS[0], window=100, review=REVIEW[0], review_pick=0, auto_continue=True,
                action="run", proxy_width=960, area_ratio=0.35, jump=0.15, validate_backward=False,
                confidence_threshold=0.5, reid_interval=0, drift_similarity=0.5, max_gap=0, min_confidence=0.5,
                margin=0.05, motion_weight=0.5, search_distance=1.0, detection_threshold=0.4,
                max_auto_runs=200) -> io.NodeOutput:
        who = "H3 Temporal Subject Tracker"
        mem = ST.parse_subject(subject, who)
        if mem is None:
            raise ValueError(f"{who}: connect 'subject' (H3 Multi-Subject Selector).")
        src = _Source(media, images, int(proxy_width), who)
        if backend == BACKENDS[0] and model is None:
            raise ValueError(f"{who}: backend = SAM 3 (core) needs 'model' - load the SAM 3 / 3.1 checkpoint with "
                             "core Load Checkpoint and connect its MODEL.")
        if backend == BACKENDS[1]:
            if masks is None:
                raise ValueError(f"{who}: backend = masks input needs 'masks' (a track of the whole clip).")
            mk = masks if masks.ndim == 3 else masks.unsqueeze(0)
            if int(mk.shape[0]) != src.n:
                raise ValueError(f"{who}: 'masks' has {int(mk.shape[0])} frames, the clip {src.n}.")
        seed0 = _resize_mask(initial_mask[0] if initial_mask.ndim == 3 else initial_mask, src.ph, src.pw)
        if BD.bbox(seed0) is None:
            raise ValueError(f"{who}: 'initial_mask' is empty - connect H3 Multi-Subject Selector's mask.")
        if backend == BACKENDS[1]:
            validate_backward = False                 # a finished track has no backward pass to compare with
        settings = {"backend": backend, "window": int(window), "proxy": [src.pw, src.ph], "area_ratio": area_ratio,
                    "jump": jump, "validate_backward": bool(validate_backward), "confidence_threshold":
                    confidence_threshold, "reid_interval": int(reid_interval), "drift_similarity": drift_similarity,
                    "max_gap": int(max_gap), "min_confidence": min_confidence, "margin": margin, "motion_weight":
                    motion_weight, "search_distance": search_distance, "detection_threshold": detection_threshold}
        key = J.fingerprint([src.identity(), mem["subject_id"], J.fingerprint(seed0[::4, ::4].tolist()), settings])[:12]
        base = _track_root(job)
        if action == "restart" and (base / key).is_dir():
            os.replace(base / key, base / f"{key}.restart{int(time.time())}")
        hist = np.asarray(mem["hist"], dtype=np.float64)
        fresh = {"subject_id": mem["subject_id"], "first_box": mem.get("box"), "first_frame_size":
                 mem.get("frame_size"), "job": J.safe_name(job), "key": key, "n": src.n,
                 "source_size": [src.w, src.h], "proxy": [src.pw, src.ph], "window": int(window), "settings": settings,
                 "status": "tracking", "cursor": 0, "seed": "", "hist_sum": hist.tolist(), "hist_n": 1,
                 "areas": [], "last_box": None, "boxes": {}, "confidence": {}, "events": [], "review": None,
                 "auto_chain": 0, "auto_token": ""}
        t = Track.open(base, key, fresh)
        if not t.s["seed"]:
            t.reseed(0, seed0)
            t.s["areas"], t.s["last_box"] = [], None
            t.event(0, "start", subject_id=mem["subject_id"], frames=src.n)
            t.save()
        hid = getattr(cls, "hidden", None)
        token = str(((getattr(hid, "extra_pnginfo", None) or {}).get("h3_track_token")) or "")
        note = ""
        if token and token != t.s.get("auto_token"):
            note = "this automatic run is stale (a newer run continued the track) - nothing was done"
            return cls._outputs(t, src, None, None, note)
        if not token:
            t.s["auto_chain"] = 0
        if t.s["status"] == "waiting review" and review != "wait":
            note = apply_review(t, src, review, int(review_pick))
        wm, lo = None, None
        if t.s["status"] == "tracking":
            if backend == BACKENDS[1]:
                def track_fn(frames, seed, _m=mk):           # the window starts at the cursor
                    a = int(t.s["cursor"])
                    return _m[a:a + int(frames.shape[0])].float().cpu()
            else:
                def track_fn(frames, seed):
                    return _TRACK(model, frames, seed)
            detect_fn = None
            if model is not None and conditioning is not None:
                def detect_fn(frame):
                    return _DETECT(model, conditioning, frame, detection_threshold)
            wm, lo = run_window(t, src, track_fn=track_fn, detect_fn=detect_fn, area_ratio=area_ratio, jump=jump,
                                validate_backward=validate_backward, confidence_threshold=confidence_threshold,
                                reid_interval=reid_interval, drift_similarity=drift_similarity, max_gap=max_gap,
                                min_confidence=min_confidence, margin=margin, motion_weight=motion_weight,
                                search_distance=search_distance)
            msg = cls._auto(t, auto_continue, int(max_auto_runs))
            note = "; ".join(x for x in (note, msg) if x)
        return cls._outputs(t, src, wm, lo, note)

    @classmethod
    def _auto(cls, t: Track, auto_continue: bool, max_auto_runs: int) -> str:
        if t.s["status"] != "tracking":
            return ""
        if os.environ.get("H3_AUTO_CONTINUE", "").strip() == "0":
            return "auto-continue is switched off (H3_AUTO_CONTINUE=0) - queue again for the next window"
        if not auto_continue:
            return "auto-continue is off - queue again for the next window"
        chain = int(t.s.get("auto_chain", 0))
        if chain >= max_auto_runs:
            return f"auto-continue stopped after {chain} runs (max_auto_runs) - queue again to continue"
        hid = getattr(cls, "hidden", None)
        prompt, uid = getattr(hid, "prompt", None), str(getattr(hid, "unique_id", None) or "")
        if not isinstance(prompt, dict) or uid not in prompt:
            return "auto-continue could not find this node in the prompt - queue again for the next window"
        nxt = json.loads(json.dumps(prompt))
        for node in nxt.values():
            if isinstance(node, dict):
                node.pop("is_changed", None)           # a copy that kept them would replay this run (D0.36)
        nxt[uid].setdefault("inputs", {}).update(review="wait", action="run")
        tok = uuid.uuid4().hex
        t.s["auto_token"], t.s["auto_chain"] = tok, chain + 1
        t.save()
        extra = dict(getattr(hid, "extra_pnginfo", None) or {}, h3_track_token=tok)
        try:
            pid = (_POSTER or HJ.post_prompt)(nxt, extra)
        except Exception as exc:  # noqa: BLE001 - said in the status, never retried in a loop
            return f"auto-continue could not queue the next window ({exc}) - queue it yourself"
        return f"queued the next window automatically (run {chain + 1}, prompt {str(pid)[:8]})"

    @classmethod
    def _outputs(cls, t: Track, src: _Source, wm, lo, note: str) -> io.NodeOutput:
        s = t.s
        n, (pw, ph) = int(s["n"]), s["proxy"]
        boxes = [[] if s["boxes"].get(str(i)) is None else _bbox_out(s["boxes"][str(i)])[0]
                 for i in range(n) if str(i) in s["boxes"]]
        conf = min([float(v) for v in s["confidence"].values()] or [1.0])
        done = sum(1 for _ in s["boxes"])
        status = f"{s['status']}: {done}/{n} frames"
        if s["status"] == "waiting review" and s.get("review"):
            status += f" - WAITING FOR REVIEW at frame {s['review']['frame']}: {s['review']['why']}. Set review = " \
                      "'continue with candidate' and review_pick = the '#' in the preview, or 'end the track here'."
        if note:
            status += f" - {note}"
        if wm is None:                                     # nothing tracked this run: show the last frames we have
            have = sorted(int(k) for k in s["boxes"])
            lo = max(0, (have[-1] if have else 0) - 7)
            hi = min(n, lo + 8)
            wm = torch.from_numpy(np.stack([_load(t.mask_path(i)) if os.path.isfile(t.mask_path(i))
                                            else np.zeros((ph, pw), np.float32) for i in range(lo, hi)]))
        frames = src.read(lo, lo + int(wm.shape[0]))
        path = [((b[0] + b[2]) / 2 / src.w, (b[1] + b[3]) / 2 / src.h) for b in
                (s["boxes"].get(str(i)) for i in range(n)) if b][-200:]
        prev = preview(frames, wm, path)
        if s["status"] == "waiting review" and s.get("review"):
            f = int(s["review"]["frame"])
            fr = src.read(f, f + 1)[0].numpy()
            cands = []
            for p in sorted((t.root / "review").glob(f"f{f:06d}_c*.png")):
                m = _load(str(p))
                b = BD.bbox(m)
                if b is not None:
                    cands.append({"index": int(p.stem.split("_c")[1]), "box": list(b), "mask": m})
            sheet = gallery(fr, cands, s["review"].get("ranking") or [], None, tile=min(ph, 256))[0]
            prev = HJ._letterbox(sheet, pw, ph)[None]
        last = s.get("last_box")
        box, size = (ST.scaled_box(last, s["proxy"], s["source_size"]), s["source_size"]) if last else \
            (s.get("first_box") or [0, 0, 0, 0], s.get("first_frame_size") or s["source_size"])
        upd = {"c2c_contract": ST.SUBJECT_CONTRACT, "c2c_contract_version": ST.SUBJECT_VERSION,
               "subject_id": s["subject_id"], "frame": max([int(k) for k in s["boxes"]] or [0]),
               "box": [round(float(v), 2) for v in box], "frame_size": list(size),
               "hist": [round(float(v), 6) for v in (t.mean_hist if t.mean_hist is not None else [])],
               "area": float(s["areas"][-1]) if s["areas"] else 0.0, "track": s["key"]}
        manifest = {"c2c_contract": TRACK_CONTRACT, "c2c_contract_version": TRACK_VERSION, "job": s["job"],
                    "key": s["key"], "folder": str(t.root), "status": s["status"], "frames": n, "tracked": done,
                    "proxy": s["proxy"], "source_size": s["source_size"], "subject_id": s["subject_id"],
                    "masks": "masks/{:06d}.png", "boxes": s["boxes"], "events": len(s["events"])}
        if s["status"] == "done":
            matte = matte_handle(t, src)
        elif s["status"] == "tracking":                  # the job is under way: what follows waits, without an error
            from comfy_execution.graph_utils import ExecutionBlocker
            matte = ExecutionBlocker(None)
        else:
            matte = HJ._blocker(f"H3 Temporal Subject Tracker: the track is waiting for review ({done}/{n} frames) - "
                                "see its status; the matte is ready when the track is done.")
        return io.NodeOutput(wm.float(), boxes, float(conf), json.dumps(s["events"][-50:]), json.dumps(upd), status,
                             prev, json.dumps(manifest), matte)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_MultiSubjectSelector": MiniMaxH3_MultiSubjectSelector,
    "MiniMaxH3_SubjectReIdentification": MiniMaxH3_SubjectReIdentification,
    "MiniMaxH3_TemporalSubjectTracker": MiniMaxH3_TemporalSubjectTracker,
}
