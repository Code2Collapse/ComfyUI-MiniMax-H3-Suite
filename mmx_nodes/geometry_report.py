"""MiniMaxH3_GeometryReport - where every conditioning stream lands, before a GPU second is spent (A16 Part II).

Original implementation, GPL-3.0. Reads the conditioning and the latent exactly as the sampler will get them and
reports, per stream: what it is, its encoded size, its token grid against the target's, the offset its content is
fed at (core's left-edge RoPE convention, mmx_utils/h3_geometry.py), keyframe / target grid agreement, declared vs
actual reference latent sizes, VAE centre-crop margins, audio length, and what the noise mask means once core has
pooled it onto the token grid. Diagnosis: docs/evidence/A16/offset/ROOT_CAUSE_AND_DESIGN.md (workspace).

No model, no weights, no sampling - it only reads tensors and their shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import h3_geometry as geo  # noqa: E402


def _streams(samples):
    """(video, audio) of a joint H3 latent, or (video, None) for a plain one."""
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return samples, None


def _cond_entries(conditioning) -> list[dict]:
    out = []
    for item in conditioning or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
            out.append(item[1])
    return out


def build_report(positive, latent, negative=None, bboxes=None) -> dict[str, Any]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("H3 Geometry Report: 'latent' is not a latent (no 'samples').")
    video, audio = _streams(latent["samples"])
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError(f"H3 Geometry Report: expected an H3 video latent [B, 24, T, H/16, W/16], got "
                         f"{tuple(video.shape)}.")
    lat_t, lat_h, lat_w = int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    W, H = lat_w * geo.LATENT_STRIDE, lat_h * geo.LATENT_STRIDE
    frames = geo.frame_count_for_latent_t(lat_t)
    rep: dict[str, Any] = {"target": {"latent": [lat_t, lat_h, lat_w], "pixels": [W, H], "frames": frames,
                                      "token_px": geo.TOKEN_PX, "odd_latent_edge": bool(lat_h % 2 or lat_w % 2)},
                           "streams": [], "warnings": [], "notes": []}
    if lat_h % 2 or lat_w % 2:
        rep["notes"].append(f"The target latent is {lat_w}x{lat_h}: core pads it to an even size at the bottom/right "
                            "for the 2x2 patch and crops the result back - no shift, but the last token row/column "
                            "is half padding.")
    if audio is not None:
        exp = geo.expected_audio_ticks(frames)
        got = int(audio.shape[-1])
        rep["target"]["audio_ticks"] = got
        rep["target"]["audio_ticks_expected"] = exp
        if got != exp:
            rep["warnings"].append(f"The audio latent has {got} ticks but {frames} frames need {exp} "
                                   f"(round({frames}/24*40)): picture and sound end at different times.")
    else:
        rep["warnings"].append("The latent is video-only. MiniMax H3 samples picture and sound together; wire the "
                               "joint AV latent (Source AV Latent, Empty MiniMax H3 AV Latent).")

    # noise masks, as core will pool them
    nm = latent.get("noise_mask")
    if nm is not None:
        v_mask, a_mask = _streams(nm)
        try:
            vm = v_mask.float()
            if vm.ndim == 5:                       # [B, C, T, h, w] - the latent's own shape
                vm = vm[0].amax(dim=0)
            elif vm.ndim == 4:                     # [T, 1, h, w] (Set Latent Noise Mask style) or [1, T, h, w]
                vm = vm[:, 0] if vm.shape[1] == 1 else vm[0]
            elif vm.ndim == 2:
                vm = vm[None]
            if vm.shape[0] not in (1, lat_t):
                rep["notes"].append(f"The video noise mask has {vm.shape[0]} frames for {lat_t} latent rows: the "
                                    "sampler resamples it in time before core pools it (H3 Mask To Latent Space "
                                    "reduces it on H3's own frame grouping instead).")
            if vm.shape[-2:] != (lat_h, lat_w):
                rep["notes"].append(f"The video noise mask is {vm.shape[-1]}x{vm.shape[-2]}, not the latent's "
                                    f"{lat_w}x{lat_h}: the sampler resizes it first (bilinear), then core pools it.")
            else:
                cov = geo.mask_token_coverage(vm)
                rep["mask"] = {"video": cov}
                rep["notes"].append(f"Video noise mask on the token grid: {cov['regenerated_pct']}% of tokens "
                                    f"regenerated, {cov['kept_pct']}% kept, {cov['partial']} partial (1 = generate).")
        except Exception as exc:  # noqa: BLE001 - a report must not die on an odd mask
            rep["notes"].append(f"Video noise mask not read ({type(exc).__name__}: {exc}).")
        if a_mask is not None and audio is not None:
            try:
                am = a_mask.float()
                keep = float((am <= 1e-6).float().mean()) * 100.0
                rep.setdefault("mask", {})["audio_kept_pct"] = round(keep, 2)
                rep["notes"].append(f"Audio noise mask: {keep:.1f}% of ticks kept.")
            except Exception as exc:  # noqa: BLE001
                rep["notes"].append(f"Audio noise mask not read ({type(exc).__name__}: {exc}).")

    def scan(conditioning, which: str):
        for entry in _cond_entries(conditioning):
            for k, kf in enumerate(entry.get("minimax_keyframes") or []):
                s = {"stream": f"{which} keyframe {k + 1}", "frame": kf.get("resolved_frame_index")}
                z = kf.get("latent")
                if z is not None:
                    s["latent"] = list(z.shape[2:])
                    if tuple(z.shape[-2:]) != (lat_h, lat_w):
                        rep["warnings"].append(
                            f"{which} keyframe {k + 1} is {z.shape[-1] * 16}x{z.shape[-2] * 16} px but the target is "
                            f"{W}x{H}: keyframes are packed on the TARGET's grid, so its tokens no longer line up with "
                            "the frame it anchors. Build guides from this latent (Add Guide for MiniMax H3) or size "
                            "first/last frames to the target.")
                        s["matches_target_grid"] = False
                    else:
                        s["matches_target_grid"] = True
                rep["streams"].append(s)
            for k, blk in enumerate(entry.get("minimax_refs") or []):
                kind = blk.get("kind")
                s = {"stream": f"{which} reference {k + 1} ({kind})", "kind": kind}
                z = blk.get("latent")
                dh, dw = blk.get("latent_h"), blk.get("latent_w")
                if z is not None and dh is not None:
                    s["latent_declared"] = [dh, dw]
                    s["latent_actual"] = list(z.shape[-2:])
                    if (int(dh), int(dw)) != (int(z.shape[-2]), int(z.shape[-1])):
                        rep["warnings"].append(
                            f"{which} reference {k + 1}: its block says {dw}x{dh} latent cells but the encoded latent "
                            f"is {z.shape[-1]}x{z.shape[-2]}. Core packs positions for the declared size, so the "
                            "reference tokens and their coordinates disagree.")
                if dh is not None and dw is not None:
                    px = (int(dw) * geo.LATENT_STRIDE, int(dh) * geo.LATENT_STRIDE)
                    s["pixels"] = list(px)
                    s["rope_extent"] = geo.rope_extent(int(dh), int(dw))
                    if kind in ("video", "video_audio"):
                        try:
                            off = geo.stream_offset_px((W, H), px)
                            s["offset_if_same_shot"] = off
                            if not off["same_grid"]:
                                scale = geo.plate_scale(_first_box(bboxes), (W, H))
                                tail = (f" (about {off['max_abs_px'] * scale:.1f} px in the original plate)"
                                        if scale else "")
                                rep["warnings"].append(
                                    f"{which} reference {k + 1} (video, {px[0]}x{px[1]}): if it shows the same shot "
                                    f"as the target (source, pose, depth) it is {geo.describe_offset(off)}{tail}. "
                                    "Size it to the target grid (H3 Image + Reference to Video: ref_video_fit = "
                                    "'target grid (zero offset)').")
                        except geo.H3GeometryError as exc:
                            s["offset_if_same_shot"] = str(exc)
                    elif kind == "image":
                        ext, text = s["rope_extent"], geo.rope_extent(lat_h, lat_w)
                        if ext["w"][0] < text["w"][0] - 4 or ext["w"][1] > text["w"][1] + 4 or \
                                ext["h"][0] < text["h"][0] - 4 or ext["h"][1] > text["h"][1] + 4:
                            rep["notes"].append(
                                f"{which} reference {k + 1} (picture, {px[0]}x{px[1]}) spans RoPE h {ext['h']} w "
                                f"{ext['w']} against the target's h {text['h']} w {text['w']}: a wide strip of several "
                                "people puts each of them over a different part of the target's coordinates. One "
                                "<Picture> per reference avoids that.")
                rep["streams"].append(s)

    scan(positive, "positive")
    if negative is not None:
        scan(negative, "negative")
    g = latent.get("mmx_geometry")
    if isinstance(g, dict):
        rep["source_geometry"] = g
    if bboxes is not None:
        b = _first_box(bboxes)
        if b:
            rep["crop_box"] = {k: b.get(k) for k in ("x", "y", "width", "height")}
            sx, sy = float(b["width"]) / W, float(b["height"]) / H
            rep["crop_box"]["target_to_plate_scale"] = [round(sx, 5), round(sy, 5)]
            if abs(sx - sy) > 1e-3:
                rep["notes"].append(f"The crop box is {b['width']}x{b['height']} and the target {W}x{H}: the paste "
                                    f"back scales x by {sx:.4f} and y by {sy:.4f} (an exact inverse of the crop's "
                                    "resize, not a shift).")
    return rep


def _first_box(bboxes):
    if bboxes is None:
        return None
    if isinstance(bboxes, dict):
        return bboxes
    if isinstance(bboxes, list) and bboxes:
        f = bboxes[0]
        if isinstance(f, list) and f and isinstance(f[0], dict):
            return f[0]
        if isinstance(f, dict):
            return f
    return None


def report_text(rep: dict[str, Any]) -> str:
    t = rep["target"]
    lines = [f"Target: {t['pixels'][0]}x{t['pixels'][1]} px, {t['frames']} frames ({t['latent'][0]} latent rows), "
             f"tokens {t['token_px']} px" + (f", audio {t['audio_ticks']} ticks" if "audio_ticks" in t else "")]
    for s in rep["streams"]:
        bits = [s["stream"]]
        if "pixels" in s:
            bits.append(f"{s['pixels'][0]}x{s['pixels'][1]} px")
        if "latent" in s:
            bits.append(f"latent {s['latent']}")
        if s.get("matches_target_grid") is True:
            bits.append("on the target grid")
        off = s.get("offset_if_same_shot")
        if isinstance(off, dict):
            bits.append("offset 0" if off["same_grid"] else f"offset {off['dx_mean_px']:+.1f}/{off['dy_mean_px']:+.1f} px")
        lines.append("  - " + ", ".join(bits))
    if rep["warnings"]:
        lines.append("Problems:")
        lines += ["  ! " + w for w in rep["warnings"]]
    if rep["notes"]:
        lines.append("Notes:")
        lines += ["  - " + n for n in rep["notes"]]
    if not rep["warnings"]:
        lines.append("No geometry problems found.")
    return "\n".join(lines)


class MiniMaxH3_GeometryReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_GeometryReport",
            display_name="H3 Geometry Report",
            category="MiniMax H3/Diagnostics",
            description=(
                "Before sampling: where each reference, guide and mask lands on H3's token grid. Wire the positive "
                "conditioning and the latent the sampler gets. It says, in pixels, how far a reference video that "
                "shows the same shot is from the output's own tokens, whether guides share the output's grid, "
                "whether picture and sound have the same length, and what the noise mask keeps once ComfyUI has "
                "pooled it. Reads shapes only - no model, instant."),
            inputs=[
                io.Conditioning.Input("positive", tooltip="The conditioning the guider / sampler gets."),
                io.Latent.Input("latent", tooltip="The latent the sampler starts from (with its noise mask)."),
                io.Conditioning.Input("negative", optional=True, tooltip="Optional: checked the same way."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional crop boxes from Subject Crop: offsets are also given in "
                                             "original-plate pixels."),
            ],
            outputs=[
                io.String.Output(display_name="report", tooltip="Plain-English findings."),
                io.String.Output(display_name="report_json", tooltip="The same, as JSON, for tools and tests."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, positive, latent, negative=None, bboxes=None) -> io.NodeOutput:
        rep = build_report(positive, latent, negative, bboxes)
        text = report_text(rep)
        return io.NodeOutput(text, json.dumps(rep, default=str), ui={"text": [text]})


__all__ = ["MiniMaxH3_GeometryReport", "build_report", "report_text"]
