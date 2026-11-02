"""Character swap / body double nodes (owner A21 Rev 3 Parts 5, 10, 13; ledger L9.78).

H3 Temporal Deflicker and H3 Temporal Matte Chatter Corrector, on the finishing algorithms in mmx_utils/body_double.py
(ported from open-source-higgsfield, Apache-2.0). Original nodes, GPL-3.0. Design: docs/evidence/A21/phase2/
DESIGN_L9.78_character_swap.md in the workspace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import body_double as BD  # noqa: E402


class MiniMaxH3_TemporalDeflicker(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TemporalDeflicker", display_name="H3 Temporal Deflicker", category="MiniMax H3/Finish",
            description="Removes the fine frame-to-frame boil of a generated sequence where nothing moves (fine "
                        "detail the three-frame median does not share), and leaves real motion alone: where the "
                        "difference is coherent - an arm, an ear moving faster than its own width - the filter fades "
                        "out. Optionally only inside a mask (e.g. the regenerated subject). The first and last "
                        "frames pass through.",
            inputs=[io.Image.Input("images"),
                    io.Float.Input("amount", default=0.9, min=0.0, max=1.0, step=0.05,
                                   tooltip="How much of the boil is removed."),
                    io.Int.Input("radius", default=0, min=0, max=64,
                                 tooltip="Box radius in pixels that separates boil (fine) from motion (coarse). "
                                         "0 = 8 px at 704 rows, scaled with the height."),
                    io.Mask.Input("mask", optional=True,
                                  tooltip="Deflicker only here (1 = deflicker). Outside, the frames are unchanged."),
                    io.Float.Input("motion_low", default=4.0, min=0.0, max=64.0, step=0.5, advanced=True,
                                   tooltip="Coarse change (0-255 scale) below which the filter works fully."),
                    io.Float.Input("motion_high", default=8.0, min=0.5, max=128.0, step=0.5, advanced=True,
                                   tooltip="Coarse change (0-255 scale) above which the filter is off.")],
            outputs=[io.Image.Output("images"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, images, amount=0.9, radius=0, mask=None, motion_low=4.0, motion_high=8.0) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if float(motion_high) <= float(motion_low):
            raise ValueError("H3 Temporal Deflicker: motion_high must be above motion_low.")
        n = int(images.shape[0])
        if n < 3:
            return io.NodeOutput(images, json.dumps({"frames": n, "note": "fewer than 3 frames: nothing to do"}))
        frames = (images[i, ..., :3].float().cpu().numpy() for i in range(n))
        out = np.stack(list(BD.deflicker(frames, amount=float(amount), radius=int(radius) or None,
                                         lo=float(motion_low) / 255.0, hi=float(motion_high) / 255.0)))
        res = torch.from_numpy(out).to(images.dtype)
        if mask is not None:
            m = (mask if mask.ndim == 3 else mask[..., 0]).float()
            m = m.expand(n, -1, -1) if m.shape[0] == 1 else m[:n]
            if tuple(m.shape[1:3]) != tuple(images.shape[1:3]):
                raise ValueError(f"H3 Temporal Deflicker: the mask is {tuple(m.shape[1:3])} and the images "
                                 f"{tuple(images.shape[1:3])} - give a mask of the same size.")
            m = m[..., None].to(res.dtype)
            res = images[..., :3].to(res.dtype) * (1 - m) + res * m
        if images.shape[-1] > 3:
            res = torch.cat([res, images[..., 3:]], dim=-1)
        change = (res[..., :3] - images[..., :3]).abs().mean(dim=(1, 2, 3))
        rep = {"frames": n, "mean_change": round(float(change.mean()), 6), "max_change": round(float(change.max()), 6),
               "radius": int(radius) or max(2, round(int(images.shape[1]) * 8 / 704))}
        return io.NodeOutput(res, json.dumps(rep))


class MiniMaxH3_TemporalMatteChatterCorrector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TemporalMatteChatterCorrector", display_name="H3 Temporal Matte Chatter Corrector",
            category="MiniMax H3/Finish",
            description="Closes short dropouts in a mask sequence: a pixel stays covered while the mask covered it "
                        "within 'window' frames before AND after, so a segmenter losing the hair, a sleeve or the "
                        "whole subject for a few frames no longer pastes the plate back over the new cast (it reads "
                        "as flicker). A pixel the subject really leaves stays uncovered. With bboxes (crop boxes) the "
                        "masks are judged in plate space, where the set holds still while a follow-cam crop moves.",
            inputs=[io.Mask.Input("masks"),
                    io.Int.Input("window", default=24, min=1, max=240,
                                 tooltip="Frames a dropout may last and still be closed."),
                    io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                         tooltip="Crop boxes (Subject Crop) when the masks are crop-space."),
                    io.Int.Input("plate_width", default=0, min=0, max=16384,
                                 tooltip="With bboxes: the plate's width (0 = the right edge of the boxes)."),
                    io.Int.Input("plate_height", default=0, min=0, max=16384,
                                 tooltip="With bboxes: the plate's height (0 = the bottom edge of the boxes)."),
                    io.Boolean.Input("bridge_take_ends", default=False,
                                     tooltip="Also close a dropout at the very start or end of the take. Only for a "
                                             "subject that holds still: a moving subject's next 'window' frames of "
                                             "path would be painted into the first frames.")],
            outputs=[io.Mask.Output("masks"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, masks, window=24, bboxes=None, plate_width=0, plate_height=0,
                bridge_take_ends=False) -> io.NodeOutput:
        m = masks if masks.ndim == 3 else masks.unsqueeze(0)
        n, h, w = (int(v) for v in m.shape)
        if bboxes is not None:
            rects = _rects(bboxes, n)
            pw = int(plate_width) or int(max(r[0] + r[2] for r in rects))
            ph = int(plate_height) or int(max(r[1] + r[3] for r in rects))
        else:
            rects, pw, ph = [[0, 0, w, h]] * n, w, h
        arrays = [m[i].float().cpu().numpy() for i in range(n)]
        out = np.stack(list(BD.steady_masks(arrays, rects, (pw, ph), gap=int(window), scale=0.25 if bboxes is not None
                                            else 1.0, ends=bool(bridge_take_ends))))
        res = torch.from_numpy(out).clamp(0, 1).to(masks.dtype)
        added = (res - m.float()).clamp_min(0).flatten(1).mean(1)
        bridged = [i for i in range(n) if float(added[i]) > 1e-4]
        rep = {"frames": n, "window": int(window), "frames_bridged": bridged[:200],
               "mean_added": round(float(added.mean()), 6), "space": "plate (bboxes)" if bboxes is not None else "mask"}
        return io.NodeOutput(res, json.dumps(rep))


def _rects(bboxes, n: int) -> list[list[float]]:
    """BOUNDING_BOX (one box, or a per-frame list of lists, as Subject Crop gives) -> [x, y, w, h] per frame."""
    if isinstance(bboxes, dict):
        frames = [bboxes] * n
    else:
        frames = [f[0] if isinstance(f, list) and f else f for f in bboxes]
        if len(frames) == 1:
            frames = frames * n
    if len(frames) < n:
        raise ValueError(f"H3 Temporal Matte Chatter Corrector: {len(frames)} boxes for {n} masks.")
    return [[float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])] for b in frames[:n]]


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_TemporalDeflicker": MiniMaxH3_TemporalDeflicker,
    "MiniMaxH3_TemporalMatteChatterCorrector": MiniMaxH3_TemporalMatteChatterCorrector,
}
