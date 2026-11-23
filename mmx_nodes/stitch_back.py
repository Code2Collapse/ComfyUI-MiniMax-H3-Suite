"""N5 — H3 Stitch Back."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.affine_transform import affine_crop_batch, inverse_affine_theta
from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.feather_composite import (
    gaussian_blur_mask,
    mask_confined_blend,
    subject_region_mask,
)
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


class MiniMaxH3_StitchBack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_StitchBack",
            display_name="H3 Stitch Back",
            category="MiniMax H3/Spine",
            description=(
                "Paste crops back using the exact H3_TRANSFORM from Track + Crop. "
                "Inverse affine warp, feather in SOURCE pixels (default 32px = one "
                "latent token), mask-confined composite. Pixels outside the paste "
                "mask are the literal plate tensor (invariant 5)."
            ),
            inputs=[
                io.Image.Input("plate", tooltip="Full-size source frames the crops are pasted onto (same clip "
                                                "Track + Crop received). Pixels outside the paste mask are "
                                                "returned unchanged (RGB only)."),
                io.Image.Input("crops", tooltip="Generated crops at the transform's canvas size, one per frame. A "
                                                "batch the VAE cut to a multiple of 16 is re-centred onto the "
                                                "canvas with plate margins instead of stretched."),
                H3TransformType.Input("transform", tooltip="The H3_TRANSFORM from H3 Track + Crop, unchanged: "
                                                           "per-frame crop boxes used for the inverse warp, plus "
                                                           "subject rectangles and detection weights."),
                io.Mask.Input(
                    "paste_mask",
                    optional=True,
                    tooltip="Optional per-frame mask in CANVAS space.",
                ),
                io.Combo.Input(
                    "paste_region",
                    options=["subject", "subject_ellipse", "full_crop"],
                    default="subject", extra_dict={"c2c_hide_if_linked": ["paste_mask"]},
                    tooltip="Area of each crop pasted back when no paste_mask is wired. subject: the tracked "
                            "subject rectangle; subject_ellipse: the ellipse inside it; full_crop: the whole "
                            "crop. Edges are feathered by feather_px.",
                ),
                io.Int.Input("mask_dilation", default=8, min=0, max=256, step=2,
                             tooltip="Canvas pixels the paste area is grown by before feathering (subject "
                                     "rectangle/ellipse, or a max-filter on paste_mask). Not used by full_crop. "
                                     "0 = no growth."),
                io.Int.Input("feather_px", default=32, min=0, max=256, step=2,
                             tooltip="Gaussian feather of the paste edge, in SOURCE pixels (converted to canvas "
                                     "pixels per 8-frame chunk; kernel 2N+1). 32 = one H3 latent token. "
                                     "0 = hard edge."),
                io.Float.Input("colour_match", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="0-1: per frame and channel, shift the crop's mean and standard deviation "
                                       "inside the paste mask toward the plate's. 0 = crop colours as generated; "
                                       "1 = full match.",
                               extra_dict={"c2c_section": "Blend"}),
                io.Float.Input("blend", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="0-1 opacity of the pasted crop over the plate (multiplies the paste mask). "
                                       "0 = output is the plate.",
                               extra_dict={"c2c_section": "Blend"}),
                io.Combo.Input(
                    "undetected_frames",
                    options=["fade_out", "skip", "composite_anyway"],
                    default="fade_out", extra_dict={"c2c_section": "Blend"},
                    tooltip="Frames where Track + Crop found no subject. fade_out: paste opacity follows the "
                            "detection rate over a 9-frame window (ramps to 0); skip: those frames keep the plate; "
                            "composite_anyway: paste every frame.",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Plate frames (RGB) with the crops warped back and composited; "
                                                  "pixels outside the composite mask are the plate unchanged."),
                io.Mask.Output("composite_mask", tooltip="Per-frame MASK at source size, 1 = fully the pasted crop, "
                                                         "fractional on the feather, 0 = plate; includes blend and "
                                                         "detection weights."),
                io.String.Output("report", tooltip="Plain-text summary: frames stitched, planner mode, feather in "
                                                   "source and canvas pixels, any VAE 16 px cut recovery."),
            ],
        )

    @staticmethod
    def _recover_vae_cut(plate, crops, boxes, cw: int, ch: int, b: int):
        """The H3 VAE centre-crops each side to a multiple of 16; a canvas off that grid decodes smaller and the
        inverse warp stretched it back (an offset of up to 7 px). Rebuild the canvas instead: the decoded pixels at
        core's offset, the cut margins from the plate at the same crop (zero-offset plan, Z1)."""
        kw, kh, left, top = cw // 16 * 16, ch // 16 * 16, (cw % 16) // 2, (ch % 16) // 2
        got = (int(crops.shape[2]), int(crops.shape[1]))
        if got == (cw, ch) or got != (kw, kh):
            return crops, ""
        fill = affine_crop_batch(plate[:b, ..., :3], boxes[:b], cw, ch).to(device=crops.device, dtype=crops.dtype)
        fill[:, top:top + kh, left:left + kw, :] = crops[:b, ..., :3]
        return fill, (f"decoded crops were {kw}x{kh} (the VAE cut the {cw}x{ch} canvas to a multiple of 16): put "
                      f"back at +{left},+{top} with the margins from the plate - no stretch, no offset\n")

    @classmethod
    def execute(
        cls,
        plate,
        crops,
        transform: H3Transform,
        paste_mask=None,
        paste_region="subject",
        mask_dilation=8,
        feather_px=32,
        colour_match=1.0,
        blend=1.0,
        undetected_frames="fade_out",
    ):
        if plate.ndim == 3:
            plate = plate.unsqueeze(0)
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)

        boxes = list(transform.boxes)
        cw, ch = transform.canvas
        w, h = transform.src_size
        b = min(plate.shape[0], crops.shape[0], len(boxes))
        crops, vae_note = cls._recover_vae_cut(plate, crops, boxes, int(cw), int(ch), b)

        if undetected_frames == "composite_anyway":
            weights = None
        elif undetected_frames == "skip":
            weights = [1.0 if d else 0.0 for d in transform.detected]
        else:
            weights = list(transform.weights)

        dev = plate.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = plate.device

        dt = plate.dtype

        def _work(dev):
            out = plate[..., :3].clone()
            composite_masks = torch.zeros((b, h, w), dtype=torch.float32, device=dev)

            for c0 in range(0, b, 8):
                c1 = min(c0 + 8, b)
                n = c1 - c0
                bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
                f_can = max(1, int(round(float(feather_px) * (ch / max(bh_mid, 1.0)))))

                if paste_mask is not None:
                    mk = paste_mask[c0:c1].to(dev).float()
                    if mk.ndim == 2:
                        mk = mk.unsqueeze(0)
                    if mk.shape[-2:] != (ch, cw):
                        mk = F.interpolate(
                            mk.unsqueeze(1), size=(ch, cw), mode="bilinear", align_corners=False
                        )
                    else:
                        mk = mk.unsqueeze(1)
                    if mask_dilation > 0:
                        k = 2 * int(mask_dilation) + 1
                        mk = F.max_pool2d(mk, k, stride=1, padding=k // 2)
                    mask_can = gaussian_blur_mask(mk, f_can).clamp(0, 1)
                elif paste_region == "full_crop":
                    one = torch.ones((1, 1, ch, cw), device=dev)
                    mask_can = gaussian_blur_mask(one, f_can).expand(n, 1, ch, cw)
                else:
                    rects = transform.subject_rect or [
                        (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5)
                    ] * b
                    mask_can = torch.cat(
                        [
                            subject_region_mask(
                                ch,
                                cw,
                                rects[i] if i < len(rects) else rects[-1],
                                int(mask_dilation),
                                f_can,
                                paste_region == "subject_ellipse",
                                dev,
                            )
                            for i in range(c0, c1)
                        ],
                        dim=0,
                    )

                th = inverse_affine_theta(boxes[c0:c1], w, h, dev)
                grid = F.affine_grid(th, (n, 3, h, w), align_corners=False)
                patch_can = crops[c0:c1, ..., :3].to(dev).movedim(-1, 1).float()
                warped = F.grid_sample(
                    patch_can, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                warped = warped.movedim(1, -1)

                m_src = F.grid_sample(
                    mask_can, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                m_src = m_src.movedim(1, -1).clamp(0, 1)

                wv = torch.full((n, 1, 1, 1), float(blend), device=dev)
                if weights is not None:
                    for j, i in enumerate(range(c0, c1)):
                        if i < len(weights):
                            wv[j] *= float(weights[i])
                mm_ = m_src * wv

                base = out[c0:c1].to(dev).float()
                chunk_out = mask_confined_blend(base, warped, mm_, colour_match=float(colour_match))
                out[c0:c1] = chunk_out.to(out.device, dt)
                composite_masks[c0:c1] = mm_.squeeze(-1)

            return out, composite_masks.to(plate.device), f_can

        out, composite_masks, f_can = run_with_cpu_fallback(
            _work, device=dev, label="MiniMaxH3_StitchBack"
        )

        if not torch.isfinite(out).all():
            raise RuntimeError("H3 Stitch Back: output contains non-finite values.")

        report = (
            f"{vae_note}"
            f"stitched {b} frames using H3_TRANSFORM planner={transform.planner_mode}\n"
            f"feather_px={feather_px} (source) canvas_feather~{f_can}px\n"
            f"exterior pixels are literal plate (torch.where invariant)."
        )
        return io.NodeOutput(out, composite_masks, report)
