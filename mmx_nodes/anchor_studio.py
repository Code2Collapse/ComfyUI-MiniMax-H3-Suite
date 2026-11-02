"""H3 Anchor Studio, H3 Anchor Approval and H3 Prop Consistency Checker (owner A21 Rev 3 Part 10; ledger L9.78
slice 4).

The anchor is the first frame of the take repainted with the new character: every later segment is judged against
it. The CANDIDATES come from the user's own H3 image path in the graph (GPU); these nodes compare them, keep the
person's mask, record the approval in the job and in the Reference Bank (rejected candidates are never used), and stop
a take whose references hold a prop the performer does not. Detection is ComfyUI core's SAM 3 / 3.1, called at run
time. Logic: mmx_utils/anchors.py (ported from open-source-higgsfield, Apache-2.0). Original nodes, GPL-3.0.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import anchors as AN  # noqa: E402
from mmx_utils import body_double as BD  # noqa: E402
from mmx_utils import h3_job as J  # noqa: E402

from . import h3_job_nodes as HJ  # noqa: E402
from .review_package import _label  # noqa: E402
from .segment_io import job_folder  # noqa: E402

CATEGORY = "MiniMax H3/Subject"
CANDIDATES = "h3.anchor_candidates"


def _sha(img: torch.Tensor) -> str:
    q = HJ._q16(img)
    return hashlib.sha256(q.tobytes() + str(q.shape).encode()).hexdigest()


def _fit(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if (int(img.shape[0]), int(img.shape[1])) == (h, w):
        return img[..., :3].float()
    return F.interpolate(img[..., :3].float().movedim(-1, 0)[None], size=(h, w), mode="bilinear",
                         align_corners=False)[0].movedim(0, -1)


def _fit_mask(m, h: int, w: int) -> np.ndarray:
    a = BD._np(m)
    if a.ndim == 3:
        a = a[0]
    return a if a.shape == (h, w) else BD.resize2d(a, h, w)


def _text_cond(clip, text: str):
    """CLIP Text Encode's own conditioning for ``text`` (core's SAM 3 encoder reads 'name:N' as up to N objects)."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def _core_detect_text(model, conditioning, image: torch.Tensor, threshold: float) -> list[torch.Tensor]:
    """core SAM3_Detect on one image [H, W, 3] with a text conditioning -> one mask per object found."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    m = SAM3_Detect.execute(model, image[None, ..., :3], conditioning=conditioning, threshold=float(threshold),
                            individual_masks=True).args[0]
    return [m[i].float().cpu() for i in range(int(m.shape[0]))]


_DETECT_TEXT = _core_detect_text                             # tests replace it


# ── H3 Anchor Studio ─────────────────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_AnchorStudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AnchorStudio", display_name="H3 Anchor Studio", category=CATEGORY,
            description="Compares anchor candidates (the take's first frame repainted with the new character, made by "
                        "your H3 image path) with the source frame: a contact sheet with each candidate's numbers, "
                        "the new character's mask in each (the person standing where the performer stands, found by "
                        "core SAM 3), its colour match to the source subject and to the references, and a difference "
                        "map. Approve one with H3 Anchor Approval. Colour match is clothing and body colour, not a "
                        "face comparison.",
            inputs=[io.Image.Input("source", tooltip="The source frame (or crop) the candidates repaint."),
                    io.Image.Input("candidates", tooltip="The anchor candidates (a batch)."),
                    io.Mask.Input("source_mask", optional=True,
                                  tooltip="The performer in the source frame (H3 Multi-Subject Selector's mask)."),
                    io.Model.Input("model", optional=True, tooltip="SAM 3 / 3.1 MODEL: finds the person in each "
                                                                   "candidate."),
                    io.Conditioning.Input("conditioning", optional=True, tooltip="The text 'person:20' (CLIP Text Encode with the SAM 3 checkpoint's CLIP; ':20' lets SAM 3 return up to 20 people - plain 'person' returns only one)."),
                    io.Mask.Input("candidate_masks", optional=True,
                                  tooltip="Or the person in each candidate, one mask per candidate."),
                    io.Image.Input("references", optional=True,
                                   tooltip="The new character's references (colour match)."),
                    io.Mask.Input("reference_masks", optional=True,
                                  tooltip="The character in each reference (default: found by SAM 3 when model + "
                                          "conditioning are connected, else the whole image)."),
                    io.String.Input("prompt", default="", multiline=True,
                                    tooltip="The anchor prompt: kept in the metadata with the approval."),
                    io.Float.Input("detection_threshold", default=0.4, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Float.Input("min_overlap", default=0.3, min=0.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="The candidate's person must cover this share of the performer's area; "
                                           "someone merely touching the spot does not count."),
                    io.Int.Input("tile", default=256, min=96, max=1024, advanced=True)],
            outputs=[io.Image.Output("candidates", tooltip="At the source size."),
                     io.Mask.Output("masks", tooltip="The new character in each candidate."),
                     io.Image.Output("sheet", tooltip="Source + candidates, numbered, with their numbers."),
                     io.Image.Output("difference", tooltip="|candidate - source| x 4 per candidate."),
                     io.String.Output("metadata", tooltip="h3.anchor_candidates: for H3 Anchor Approval.")],
        )

    @classmethod
    def execute(cls, source, candidates, source_mask=None, model=None, conditioning=None, candidate_masks=None,
                references=None, reference_masks=None, prompt="", detection_threshold=0.4, min_overlap=0.3,
                tile=256) -> io.NodeOutput:
        who = "H3 Anchor Studio"
        src = (source if source.ndim == 4 else source[None])[0, ..., :3].float().cpu()
        h, w = int(src.shape[0]), int(src.shape[1])
        cands = candidates if candidates.ndim == 4 else candidates[None]
        n = int(cands.shape[0])
        fitted = torch.stack([_fit(cands[k].cpu(), h, w) for k in range(n)])
        where = _fit_mask(source_mask, h, w) if source_mask is not None else None
        given = None
        if candidate_masks is not None:
            cm = candidate_masks if candidate_masks.ndim == 3 else candidate_masks[None]
            if int(cm.shape[0]) != n:
                raise ValueError(f"{who}: {int(cm.shape[0])} candidate masks for {n} candidates - one per candidate.")
            given = [_fit_mask(cm[k], h, w) for k in range(n)]
        src_hist = BD.color_hist(src.numpy(), where if where is not None else np.ones((h, w), np.float32))
        ref_hist, ref_how = None, None
        if references is not None:
            r = references if references.ndim == 4 else references[None]
            rm = None
            if reference_masks is not None:
                rm = reference_masks if reference_masks.ndim == 3 else reference_masks[None]
                if int(rm.shape[0]) != int(r.shape[0]):
                    raise ValueError(f"{who}: {int(rm.shape[0])} reference masks for {int(r.shape[0])} references.")
            hs, hows = [], []
            for k in range(int(r.shape[0])):
                img = r[k, ..., :3].float().cpu()
                rh, rw = int(img.shape[0]), int(img.shape[1])
                pm, how = np.ones((rh, rw), np.float32), "whole image"
                if rm is not None:
                    pm, how = _fit_mask(rm[k], rh, rw), "given"
                elif model is not None and conditioning is not None:
                    found = _DETECT_TEXT(model, conditioning, img, detection_threshold)
                    pick = AN.person_overlapping([_fit_mask(m, rh, rw) for m in found], None)
                    if pick is not None:
                        pm, how = _fit_mask(found[pick], rh, rw), "core SAM 3 (largest person)"
                hs.append(BD.color_hist(img.numpy(), pm if (pm > 0.5).any() else np.ones((rh, rw), np.float32)))
                hows.append(how)
            ref_hist, ref_how = np.mean(hs, axis=0), hows
        rows, masks = [], []
        for k in range(n):
            img = fitted[k]
            person, how, checked, share = None, "not checked (connect model + conditioning, or candidate_masks)", \
                False, None
            if given is not None:
                person, how, checked = given[k], "given", True
            elif model is not None and conditioning is not None:
                found = [_fit_mask(m, h, w) for m in _DETECT_TEXT(model, conditioning, img, detection_threshold)]
                pick, share = AN.person_overlap(found, where)
                checked = True
                if pick is None:
                    how = "no person found"
                elif where is not None and share < float(min_overlap):
                    how = (f"no person where the performer stands ({len(found)} found; the closest, #{pick}, covers "
                           f"{share:.0%} of the performer's area - under {float(min_overlap):.0%})")
                else:
                    person, how = found[pick], f"core SAM 3 ({len(found)} found, #{pick} covers {share:.0%})"
            m = (person > 0.5).astype(np.float32) if person is not None else np.zeros((h, w), np.float32)
            masks.append(torch.from_numpy(m))
            use = m if m.any() else np.ones((h, w), np.float32)
            hist = BD.color_hist(img.numpy(), use)
            rows.append({"index": k, "sha256": _sha(img), "person": bool(m.any()) if checked else None,
                         "person_how": how, "person_area": int(m.sum()),
                         "overlap": None if share is None else round(float(share), 4),
                         "colour_source": round(BD.hist_similarity(hist, src_hist), 4),
                         "colour_references": None if ref_hist is None else
                         round(BD.hist_similarity(hist, ref_hist), 4),
                         "mean_difference": round(float((img - src).abs().mean()), 4)})
        meta = {"c2c_contract": CANDIDATES, "c2c_contract_version": "1.0", "source_sha256": _sha(src),
                "size": [w, h], "prompt": str(prompt or ""), "references_person": ref_how, "candidates": rows,
                "time": time.time()}
        tiles = [_label(HJ._letterbox(src, int(tile), int(tile)), "source")]
        for r, k in zip(rows, range(n)):
            lab = f"#{k}"
            if r["person"] is False:
                lab += " NO PERSON"
            lab += f" src {r['colour_source']:.2f}"
            if r["colour_references"] is not None:
                lab += f" ref {r['colour_references']:.2f}"
            tiles.append(_label(HJ._letterbox(fitted[k], int(tile), int(tile)), lab))
        sheet = torch.cat(tiles, dim=1)[None]
        diff = ((fitted - src[None]).abs() * 4).clamp(0, 1)
        return io.NodeOutput(fitted, torch.stack(masks), sheet, diff, json.dumps(meta))


# ── H3 Anchor Approval ───────────────────────────────────────────────────────────────────────────────────────────

def approved_anchor(j: J.Job, source_sha: str) -> dict | None:
    """The latest approval for this source frame (a rejected candidate never counts)."""
    for a in reversed(j.m.get("anchors") or []):
        if a.get("source_sha256") == source_sha and a.get("status") == "approved":
            return a
    return None


class MiniMaxH3_AnchorApproval(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AnchorApproval", display_name="H3 Anchor Approval", category=CATEGORY,
            description="Approve or reject an anchor candidate (H3 Anchor Studio's '#'). The decision is kept in the "
                        "H3 job and the approved anchor joins the Reference Bank as identity, so later runs, H3 Subject "
                        "Memory and every segment use it; a rejected candidate is never used. With decision = review "
                        "it gives the approved anchor of this source frame; until one is approved, what needs it waits "
                        "(no error) and 'status' says what to do.",
            inputs=[io.Image.Input("candidates"), io.String.Input("metadata", force_input=True),
                    io.Mask.Input("masks", optional=True),
                    io.String.Input("bank", optional=True, force_input=True,
                                    tooltip="A Reference Bank to add the approved anchor to (role identity)."),
                    io.String.Input("job", default="h3_job", tooltip="The H3 job that keeps the decision."),
                    io.Combo.Input("decision", options=["review", "approve", "reject"], default="review"),
                    io.Int.Input("candidate", default=0, min=0, max=256),
                    io.String.Input("note", default="")],
            outputs=[io.Image.Output("anchor"), io.Mask.Output("anchor_mask"),
                     io.String.Output("bank", tooltip="The bank with the approved anchor (identity)."),
                     io.String.Output("status")],
        )

    @classmethod
    def fingerprint_inputs(cls, job="h3_job", decision="review", candidate=0, note="", **_kw):
        try:
            return f"{J.Job.open(HJ._jobs_dir(), job, create=False, recover=False).state_token()}|{decision}|" \
                   f"{candidate}|{note}"
        except Exception:  # noqa: BLE001 - a new job
            return f"new|{decision}|{candidate}|{note}"

    @classmethod
    def execute(cls, candidates, metadata, masks=None, bank=None, job="h3_job", decision="review", candidate=0,
                note="") -> io.NodeOutput:
        who = "H3 Anchor Approval"
        meta = HJ._json(metadata, CANDIDATES, who)
        cands = candidates if candidates.ndim == 4 else candidates[None]
        rows = meta["candidates"]
        if int(cands.shape[0]) != len(rows):
            raise ValueError(f"{who}: {int(cands.shape[0])} candidates but the metadata lists {len(rows)} - connect "
                             "the same H3 Anchor Studio's candidates and metadata.")
        j = J.Job.open(HJ._jobs_dir(), job, create=True, recover=False)
        j.m.setdefault("anchors", [])
        k = int(candidate)
        if decision in ("approve", "reject"):
            if not 0 <= k < len(rows):
                raise ValueError(f"{who}: there is no candidate #{k} - the candidates are 0..{len(rows) - 1}.")
            if _sha(cands[k, ..., :3].float().cpu()) != rows[k]["sha256"]:
                raise ValueError(f"{who}: candidate #{k} is not the image H3 Anchor Studio measured - connect its "
                                 "candidates output.")
            if decision == "approve" and rows[k].get("person") is False:
                raise ValueError(f"{who}: candidate #{k} has no person where the performer stands - it cannot be the "
                                 "anchor. Pick another or make new candidates.")
        latest = next((a for a in reversed(j.m["anchors"]) if a.get("source_sha256") == meta["source_sha256"]
                       and a.get("sha256") == (rows[k]["sha256"] if 0 <= k < len(rows) else None)), None)
        repeat = latest is not None and latest.get("status") == {"approve": "approved", "reject": "rejected"}.get(
            decision)                     # a decision left set is repeated by every later (auto-continued) run: once
        if decision in ("approve", "reject") and not repeat:
            entry = {"sha256": rows[k]["sha256"], "source_sha256": meta["source_sha256"], "candidate": k,
                     "status": "approved" if decision == "approve" else "rejected", "time": time.time(),
                     "note": str(note or ""), "prompt": meta.get("prompt", ""),
                     "colour_source": rows[k].get("colour_source"),
                     "colour_references": rows[k].get("colour_references")}
            if decision == "approve":
                d = os.path.join(j.root, "anchors")
                os.makedirs(d, exist_ok=True)
                entry["path"] = os.path.join(d, f"{rows[k]['sha256'][:16]}.png")
                HJ._write_png16(entry["path"], cands[k])
                if masks is not None:
                    mk = masks if masks.ndim == 3 else masks[None]
                    entry["mask_path"] = os.path.join(d, f"{rows[k]['sha256'][:16]}_mask.png")
                    HJ._write_png16(entry["mask_path"], mk[k][..., None].expand(-1, -1, 3))
            j.m["anchors"].append(entry)
            j.save()
        a = approved_anchor(j, meta["source_sha256"])
        if a is None:
            rejected = [x["candidate"] for x in j.m["anchors"] if x["source_sha256"] == meta["source_sha256"]
                        and x["status"] == "rejected"]
            msg = (f"{who}: no anchor is approved for this source frame yet" +
                   (f" (rejected: {rejected})" if rejected else "") +
                   " - look at H3 Anchor Studio's sheet, then set decision = approve and candidate = its '#'.")
            # a wait, not a fault: what needs the anchor is skipped without an error (the rest of the graph - a track
            # job, the sheet - still runs); the status says why
            from comfy_execution.graph_utils import ExecutionBlocker
            return io.NodeOutput(ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), msg)
        img = HJ._read_png16(a["path"])[None] if os.path.isfile(a["path"]) else None
        if img is None:
            raise ValueError(f"{who}: the approved anchor {a['path']} is missing - approve a candidate again.")
        mask = torch.zeros(img.shape[1:3])
        if a.get("mask_path") and os.path.isfile(a["mask_path"]):
            mask = HJ._read_png16(a["mask_path"])[..., 0]
        b = HJ._json(bank, HJ.BANK, who) if str(bank or "").strip() else None
        folder = str(job_folder(f"h3_refs/{J.safe_name(job)}_anchors", create=True, who=who))
        if b is None or not any(e.get("approved_anchor") == a["sha256"] for e in b.get("entries") or []):
            b = HJ.add_references(b, img, ["identity"], folder)
            b["entries"][-1]["approved_anchor"] = a["sha256"]
            b["fingerprint"] = J.fingerprint([[e["sha256"], e["role"]] for e in b["entries"]])[:16]
        status = (f"anchor approved: candidate #{a['candidate']} ({a['sha256'][:12]}) for this source frame, kept in "
                  f"job '{j.m['name']}'" + (f" - {a['note']}" if a.get("note") else ""))
        return io.NodeOutput(img, mask[None], json.dumps(b), status)


# ── H3 Prop Consistency Checker ──────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_PropConsistencyChecker(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_PropConsistencyChecker", display_name="H3 Prop Consistency Checker", category=CATEGORY,
            description="Stops a take whose references (or anchor, or prompt) give the new character a prop the "
                        "performer does not hold in the source - a hat, a bag, a phone. The video model takes props "
                        "from the source performance, so such a prop pops in and out. A prop the performer already "
                        "holds is allowed (also in the anchor); locked props are always allowed. Props are found by "
                        "core SAM 3 from their names.",
            inputs=[io.Image.Input("source", tooltip="The source frame."),
                    io.Mask.Input("source_mask", tooltip="The performer in it."),
                    io.Image.Input("references", tooltip="The anchor and/or the character's references."),
                    io.Mask.Input("reference_masks", tooltip="The person in each reference, one mask per image."),
                    io.Model.Input("model", tooltip="SAM 3 / 3.1 MODEL."),
                    io.Clip.Input("clip", tooltip="SAM 3 / 3.1 CLIP: encodes each prop's name."),
                    io.String.Input("props", default="hat, bag, phone, glasses, umbrella",
                                    tooltip="The props to look for, comma-separated."),
                    io.String.Input("prompt", default="", multiline=True, tooltip="The take's prompt: a prop named "
                                                                                   "in it counts too."),
                    io.String.Input("locked_props", default="", tooltip="Props that are wanted: always allowed."),
                    io.Int.Input("anchor_index", default=-1, min=-1, max=256,
                                 tooltip="Which reference is the anchor (props the performer holds are allowed in "
                                         "it); -1 = none."),
                    io.Combo.Input("policy", options=["stop", "warn"], default="stop"),
                    io.Float.Input("threshold", default=0.7, min=0.0, max=1.0, step=0.01, advanced=True,
                                   tooltip="SAM 3 score a prop needs (the reference's 0.7: fewer false props).")],
            outputs=[io.Image.Output("references", tooltip="Passed through when the check passes (policy stop)."),
                     io.String.Output("prompt", tooltip="Passed through when the check passes (policy stop)."),
                     io.Boolean.Output("ok"), io.String.Output("report"), io.Image.Output("overlay")],
        )

    @classmethod
    def execute(cls, source, source_mask, references, reference_masks, model, clip, props="", prompt="",
                locked_props="", anchor_index=-1, policy="stop", threshold=0.5) -> io.NodeOutput:
        who = "H3 Prop Consistency Checker"
        names = [x.strip() for x in str(props or "").split(",") if x.strip()]
        locked = [x.strip() for x in str(locked_props or "").split(",") if x.strip()]
        refs = references if references.ndim == 4 else references[None]
        rm = reference_masks if reference_masks.ndim == 3 else reference_masks[None]
        if int(rm.shape[0]) != int(refs.shape[0]):
            raise ValueError(f"{who}: {int(rm.shape[0])} reference masks for {int(refs.shape[0])} references - one "
                             "per image.")
        src = (source if source.ndim == 4 else source[None])[0, ..., :3].float().cpu()
        images = [src] + [refs[k, ..., :3].float().cpu() for k in range(int(refs.shape[0]))]
        persons = [source_mask[0] if source_mask.ndim == 3 else source_mask] + [rm[k] for k in range(int(rm.shape[0]))]
        conds = {n: _text_cond(clip, f"{n}:8") for n in names}          # every instance (plain = one only)
        views, overlays = [], []
        for i, (img, pm) in enumerate(zip(images, persons)):
            h, w = int(img.shape[0]), int(img.shape[1])
            person = _fit_mask(pm, h, w) > 0.5
            found = {}
            for n in names:
                ms = _DETECT_TEXT(model, conds[n], img, threshold)
                found[n] = np.any([_fit_mask(m, h, w) > 0.5 for m in ms], axis=0) if ms else np.zeros((h, w), bool)
            views.append({"name": "source" if i == 0 else ("anchor" if i - 1 == int(anchor_index) else f"reference {i - 1}"),
                          "person": person, "props": found})
            ov = img.clone()
            ov[torch.from_numpy(person)] = ov[torch.from_numpy(person)] * 0.7 + torch.tensor([0.1, 0.5, 0.9]) * 0.3
            for m in found.values():
                ov[torch.from_numpy(m)] = ov[torch.from_numpy(m)] * 0.4 + torch.tensor([1.0, 0.2, 0.2]) * 0.6
            overlays.append(_label(HJ._letterbox(ov, 256, 256), views[-1]["name"]))
        res = AN.check_props(views, names, prompt=str(prompt or ""), locked=locked,
                             anchor_view=int(anchor_index) + 1 if int(anchor_index) >= 0 else None)
        rep = {k: v for k, v in res.items() if k != "detail"}
        rep.update(props=names, locked=locked, policy=policy, checks=res["detail"])
        overlay = torch.cat(overlays, dim=1)[None]
        if res["ok"] or policy == "warn":
            if not res["ok"]:
                rep["warning"] = AN.explain(res["found"])
            return io.NodeOutput(references, prompt, bool(res["ok"]), json.dumps(rep), overlay)
        msg = f"{who}: " + AN.explain(res["found"])
        rep["stopped"] = msg
        return io.NodeOutput(HJ._blocker(msg), HJ._blocker(msg), False, json.dumps(rep), overlay)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_AnchorStudio": MiniMaxH3_AnchorStudio,
    "MiniMaxH3_AnchorApproval": MiniMaxH3_AnchorApproval,
    "MiniMaxH3_PropConsistencyChecker": MiniMaxH3_PropConsistencyChecker,
}
