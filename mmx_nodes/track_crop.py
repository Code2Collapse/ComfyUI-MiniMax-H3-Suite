"""N1 — H3 Track + Crop."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.affine_transform import affine_crop_batch
from mmx_utils.crop_planner_tv import plan_tracked_crop_tv
from mmx_utils.detection import merge_detection_sources
from mmx_utils.h3_constants import adapt_canvas
from mmx_utils.jitterless_boxes import build_jitterless_boxes, lock_anchor_size, smooth_centers
from mmx_utils.transform_types import H3Transform, H3TransformType


class MiniMaxH3_TrackCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TrackCrop",
            display_name="H3 Track + Crop",
            category="MiniMax H3/Spine",
            description=(
                "Plan a stable per-frame crop path and emit sub-pixel warped crops plus an "
                "H3_TRANSFORM spine for stitch-back. This node is a *planner*, not a detector: "
                "wire subject_mask and/or bboxes_json from an upstream detection node. "
                "Default centre path uses clip-global TV/L1 (piecewise-constant on pans); "
                "one_euro_preview is causal preview only."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input(
                    "subject_mask",
                    optional=True,
                    tooltip="Per-frame subject mask. Extents are derived per frame.",
                ),
                io.String.Input(
                    "bboxes_json",
                    default="[]",
                    multiline=True,
                    optional=True,
                    tooltip='Per-frame boxes as JSON: [[x1,y1,x2,y2], null, ...]. Works alone for tests.',
                ),
                io.Float.Input("crop_factor", default=2.5, min=1.2, max=8.0, step=0.1),
                io.Combo.Input(
                    "canvas_mode",
                    options=["manual", "auto_no_downscale", "auto_capped_768"],
                    default="manual",
                ),
                io.Int.Input("canvas_width", default=512, min=128, max=1344, step=32),
                io.Int.Input("canvas_height", default=512, min=128, max=1344, step=32),
                io.Combo.Input(
                    "planner_mode",
                    options=["tv_lp", "one_euro_preview"],
                    default="tv_lp",
                ),
                io.Float.Input("movement_cost", default=1.0, min=0.05, max=16.0, step=0.05),
                io.Boolean.Input("lock_size", default=True),
                io.Float.Input("safety_margin", default=1.12, min=1.0, max=2.0, step=0.01),
                io.Boolean.Input("smooth_detection", default=True),
                io.Float.Input("one_euro_min_cutoff", default=1.0, min=0.05, max=10.0, step=0.05),
                io.Float.Input("one_euro_beta", default=0.05, min=0.0, max=5.0, step=0.01),
            ],
            outputs=[
                io.Image.Output("crops"),
                H3TransformType.Output("transform"),
                io.Image.Output("preview"),
                io.String.Output("report"),
                io.String.Output("boxes_json"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        images,
        subject_mask=None,
        bboxes_json="[]",
        crop_factor=2.5,
        canvas_mode="manual",
        canvas_width=512,
        canvas_height=512,
        planner_mode="tv_lp",
        movement_cost=1.0,
        lock_size=True,
        safety_margin=1.12,
        smooth_detection=True,
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.05,
    ):
        h = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        extra = (
            f"{bboxes_json}|{crop_factor}|{canvas_mode}|{canvas_width}|{canvas_height}|"
            f"{planner_mode}|{movement_cost}|{lock_size}|{safety_margin}|"
            f"{smooth_detection}|{one_euro_min_cutoff}|{one_euro_beta}"
        )
        if subject_mask is not None:
            extra += "|" + hashlib.md5(subject_mask.cpu().numpy().tobytes()).hexdigest()
        return h + ":" + hashlib.md5(extra.encode()).hexdigest()

    @classmethod
    def execute(
        cls,
        images,
        subject_mask=None,
        bboxes_json="[]",
        crop_factor=2.5,
        canvas_mode="manual",
        canvas_width=512,
        canvas_height=512,
        planner_mode="tv_lp",
        movement_cost=1.0,
        lock_size=True,
        safety_margin=1.12,
        smooth_detection=True,
        one_euro_min_cutoff=1.0,
        one_euro_beta=0.05,
    ):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        b, h, w, _c = images.shape

        bx0, bx1, by0, by1, detected = merge_detection_sources(
            frame_count=b,
            bboxes_json=bboxes_json or "[]",
            subject_mask=subject_mask,
        )

        heights = np.maximum(by1 - by0 + 1.0, 1.0)
        widths = np.maximum(bx1 - bx0 + 1.0, 1.0)
        sizes_raw = np.maximum(heights, widths) * float(crop_factor)
        max_side = float(min(w, h))
        anchor = lock_anchor_size(sizes_raw, safety_margin=safety_margin, max_side=max_side)
        crop_w = crop_h = anchor if lock_size else float(np.median(sizes_raw))

        centers = np.stack(
            [(bx0 + bx1 + 1.0) * 0.5, (by0 + by1 + 1.0) * 0.5],
            axis=1,
        ).astype(np.float32)
        if smooth_detection:
            centers = smooth_centers(
                centers,
                "one_euro",
                one_euro_min_cutoff=one_euro_min_cutoff,
                one_euro_beta=one_euro_beta,
                image_diag=float((w * w + h * h) ** 0.5),
            )

        hold_mask = ~detected
        planner_note = ""

        if planner_mode == "one_euro_preview":
            boxes, _cent = build_jitterless_boxes(
                target_centers=centers,
                anchor_size=anchor,
                crop_w=crop_w,
                crop_h=crop_h,
                W=w,
                H=h,
                one_euro_min_cutoff=one_euro_min_cutoff,
                one_euro_beta=one_euro_beta,
                hold_mask=hold_mask,
            )
            planner_note = "mode=one_euro_preview (not pan-safe)"
        else:
            tv = plan_tracked_crop_tv(
                bx0=bx0,
                bx1=bx1,
                by0=by0,
                by1=by1,
                detected=detected,
                crop_w=crop_w,
                crop_h=crop_h,
                img_w=w,
                img_h=h,
                movement_cost=movement_cost,
            )
            boxes = [
                (float(tv.x[i]), float(tv.y[i]), float(tv.crop_w), float(tv.crop_h))
                for i in range(b)
            ]
            planner_note = f"mode=tv_lp success={tv.success} msg={tv.message}"

        if canvas_mode == "manual":
            cw, ch = int(canvas_width), int(canvas_height)
        else:
            need = int(np.ceil(max(b[3] for b in boxes)))
            cap = 768 if canvas_mode == "auto_capped_768" else 10_000
            need = min(max(need, 128), cap)
            cw, ch = adapt_canvas(need, need)

        crops = affine_crop_batch(images, boxes, cw, ch)
        preview = images[..., :3].clone()
        for i, (x, y, bw, bh) in enumerate(boxes):
            xi, yi = int(round(x)), int(round(y))
            wi, hi = max(4, int(round(bw))), max(4, int(round(bh)))
            xi = min(xi, w - wi)
            yi = min(yi, h - hi)
            preview[i, yi : yi + 2, xi : xi + wi, 1] = 1.0
            preview[i, yi + hi - 2 : yi + hi, xi : xi + wi, 1] = 1.0
            preview[i, yi : yi + hi, xi : xi + 2, 0] = 1.0
            preview[i, yi : yi + hi, xi + wi - 2 : xi + wi, 0] = 1.0

        weights = detected.astype(np.float64)
        if weights.sum() > 0 and len(weights) > 2:
            k = np.ones(9, dtype=np.float64) / 9.0
            pad = len(k) // 2
            padded = np.pad(weights, (pad, pad), mode="edge")
            weights = np.convolve(padded, k, mode="valid")
        weights = np.clip(weights, 0.0, 1.0)

        subject_rect = []
        for i, (x, y, bw, bh) in enumerate(boxes):
            sx = (bx0[i] - x) / max(bw, 1e-6) * cw
            sy = (by0[i] - y) / max(bh, 1e-6) * ch
            sw = (bx1[i] - bx0[i] + 1.0) / max(bw, 1e-6) * cw
            sh = (by1[i] - by0[i] + 1.0) / max(bh, 1e-6) * ch
            subject_rect.append((float(sx), float(sy), float(sw), float(sh)))

        transform = H3Transform(
            boxes=tuple(boxes),
            canvas=(int(cw), int(ch)),
            src_size=(int(w), int(h)),
            frames=int(b),
            weights=tuple(float(v) for v in weights),
            detected=tuple(bool(v) for v in detected),
            subject_rect=tuple(subject_rect),
            crop_factor=float(crop_factor),
            planner_mode=str(planner_mode),
        )

        found = int(detected.sum())
        report = (
            f"{planner_note}\n"
            f"frames={b} detected={found}/{b} canvas={cw}x{ch} crop={crop_w:.1f}px\n"
            f"lock_size={lock_size} movement_cost={movement_cost}\n"
            f"Hand off transform to H3 Stitch Back unchanged."
        )
        boxes_payload = {
            "src_size": [int(w), int(h)],
            "canvas": [int(cw), int(ch)],
            "frames": int(b),
            "boxes": [[float(x), float(y), float(bw), float(bh)] for x, y, bw, bh in boxes],
        }
        boxes_json = json.dumps(boxes_payload, separators=(",", ":"))
        # One backdrop frame only. PreviewImage writes a PNG to temp PER BATCH FRAME, so
        # passing the whole clip would spend ~1 file per frame on every execution. The
        # widget draws a single backdrop and takes the full per-frame trajectory from
        # boxes_json, which already carries every box. The `preview` SOCKET is unchanged.
        preview_ui = ui.PreviewImage(preview[:1], cls=cls).as_dict()
        return io.NodeOutput(
            crops,
            transform,
            preview,
            report,
            boxes_json,
            ui={**preview_ui, "mmx_boxes": [boxes_json]},
        )
