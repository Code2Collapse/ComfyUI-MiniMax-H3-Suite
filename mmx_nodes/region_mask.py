"""N10 — Region mask from depth horizon or painted input (R18: MASK only)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.device import run_with_cpu_fallback
from mmx_utils.region_mask import (
    REGION_MODES,
    compute_clip_global_threshold,
    feather_mask_pixel_space,
    mask_preview,
    region_mask_from_mode,
)
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None


def _validate_depth_space(
    images: torch.Tensor,
    depth: torch.Tensor,
    transform: H3Transform | None,
) -> None:
    if transform is None:
        raise ValueError(
            "depth is connected but transform is not — depth from N9 Temporal Depth is in crop space. "
            "Connect the N1 H3_TRANSFORM so this node can verify plate vs crop alignment."
        )
    plate_h, plate_w = int(images.shape[-3]), int(images.shape[-2])
    src_h, src_w = int(transform.src_size[1]), int(transform.src_size[0])
    if (plate_h, plate_w) != (src_h, src_w):
        raise ValueError(
            f"images are {plate_w}x{plate_h} but transform.src_size is {src_w}x{src_h} — "
            "wire plate frames into Region Mask, not crops."
        )
    cw, ch = transform.canvas
    d_h, d_w = int(depth.shape[-3]), int(depth.shape[-2])
    if (d_h, d_w) == (ch, cw):
        return
    if (d_h, d_w) == (src_h, src_w):
        return
    raise ValueError(
        f"depth is {d_w}x{d_h} but transform canvas is {cw}x{ch} and plate is {src_w}x{src_h} — "
        "wire plate-space depth or N9 hints at crop resolution with matching transform."
    )


class MiniMaxH3_RegionMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RegionMask",
            display_name="H3 Region Mask",
            category="MiniMax H3/Control",
            description=(
                "Produces a pixel-space region MASK for the N2→N4→N5 spine (R18). "
                "Clip-global depth-horizon split (sky=far, ground=near). "
                "Modes: sky, ground, painted only — water/atmosphere removed (use painted_mask). "
                "When depth is connected, H3_TRANSFORM is required to validate plate vs crop space."
            ),
            inputs=[
                io.Image.Input("images"),
                H3TransformType.Input("transform", optional=True),
                io.Image.Input(
                    "depth",
                    optional=True,
                    tooltip="Depth from N9 (crop space) or plate depth — transform required when connected.",
                ),
                io.Combo.Input("mode", options=list(REGION_MODES), default="sky"),
                io.Mask.Input("painted_mask", optional=True),
                io.Float.Input("horizon_bias", default=0.0, min=-1.0, max=1.0, step=0.01),
                io.Int.Input("feather_px", default=4, min=0, max=128),
            ],
            outputs=[
                io.Mask.Output("region_mask"),
                io.Image.Output("preview"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        images,
        transform=None,
        depth=None,
        mode="sky",
        painted_mask=None,
        horizon_bias=0.0,
        feather_px=4,
    ):
        h = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
        if transform is not None:
            h += ":" + transform.fingerprint()
        if depth is not None:
            h += ":" + hashlib.md5(depth.cpu().numpy().tobytes()).hexdigest()
        if painted_mask is not None:
            h += ":" + hashlib.md5(painted_mask.cpu().numpy().tobytes()).hexdigest()
        return f"{h}|{mode}|{horizon_bias}|{feather_px}"

    @classmethod
    def execute(
        cls,
        images,
        transform=None,
        depth=None,
        mode="sky",
        painted_mask=None,
        horizon_bias=0.0,
        feather_px=4,
    ) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        b, h, w, _c = images.shape

        if depth is not None:
            _validate_depth_space(images, depth, transform)

        dev = images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                dev = images.device

        chunk = 4
        mode_l = (mode or "sky").lower()

        def _work(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
            depth_on_device = depth.to(device) if depth is not None else None
            thresh, used_fallback = compute_clip_global_threshold(
                height=h,
                width=w,
                depth_sequence=depth_on_device,
                horizon_bias=float(horizon_bias),
                device=device,
            )
            masks: list[torch.Tensor] = []
            for i in range(b):
                d_plane = None
                if depth_on_device is not None:
                    di = min(i, depth_on_device.shape[0] - 1)
                    d_plane = depth_on_device[di]
                pm = None
                if painted_mask is not None:
                    pi = min(i, painted_mask.shape[0] - 1) if painted_mask.ndim == 3 else painted_mask
                    pm = pi.to(device) if isinstance(pi, torch.Tensor) else painted_mask.to(device)
                m = region_mask_from_mode(
                    mode=mode_l,
                    height=h,
                    width=w,
                    depth=d_plane,
                    painted_mask=pm,
                    horizon_bias=float(horizon_bias),
                    threshold=thresh,
                    device=device,
                )
                m = feather_mask_pixel_space(m, int(feather_px))
                masks.append(m)
            mask_b = torch.stack(masks, dim=0)
            preview = mask_preview(images[:b].to(device), mask_b)
            depth_note = (
                "depth=absent — vertical-ramp fallback, assumes level centred horizon"
                if used_fallback
                else f"clip-global threshold={thresh:.4f} from depth sequence"
            )
            note = (
                f"mode={mode_l} horizon_bias={horizon_bias} feather_px={feather_px}px "
                f"frames={b} size={w}x{h} {depth_note} — pixel MASK for N2 mask_to_token"
            )
            return mask_b, preview, note

        mask_b, preview, note = run_with_cpu_fallback(_work, device=dev, label="MiniMaxH3_RegionMask")
        return io.NodeOutput(
            mask_b.to(images.device),
            preview.to(images.dtype),
            note,
        )
