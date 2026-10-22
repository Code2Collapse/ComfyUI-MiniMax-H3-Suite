# PORTED FROM: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale@0a8d5ee nodes/tiled_diffusion.py
# (MMH3TiledDiffusion), MIT, Copyright (c) 2026 bbaudio-2025. Changes: see mmx_utils/tiled_diffusion.py.
"""H3 Tiled Diffusion - patch an H3 model so every denoising step runs tile by tile and blends the predictions."""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.tiled_diffusion import WEIGHTS, H3TiledDiffusion, describe_settings, snap_px  # noqa: E402

CATEGORY = "MiniMax H3/Sampling"


class MiniMaxH3_TiledDiffusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TiledDiffusion",
            display_name="H3 Tiled Diffusion (experimental)",
            category=CATEGORY,
            description=(
                "Experimental. Patches an H3 model so every denoising step splits the frame into tiles, runs the model "
                "once per tile and blends the predictions. All tiles share the same sigmas and noise, so no overlap "
                "has to be frozen and no seam repaired. Each tile keeps its global frame positions, so it predicts "
                "what the full frame would. Feed the patched model to any H3 sampler, or to H3 Split Upscale with "
                "spatial_split_param unconnected. Masks and core's H3 Fun ControlNet are split per tile; the audio is "
                "whole in every tile. Costs more time than one full-frame call; it bounds memory to one tile."),
            search_aliases=["tiled diffusion", "multidiffusion", "mixture of diffusers", "h3 tiling", "tiled sampler"],
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 diffusion model to patch."),
                io.Combo.Input("method", options=list(WEIGHTS), default="trapezoid",
                               tooltip="Blend weight per tile. trapezoid: full weight inside, linear ramps over the "
                                       "overlaps (neighbours sum to 1). gaussian: Mixture of Diffusers, each tile "
                                       "dominates its centre. uniform: MultiDiffusion, plain averaging."),
                io.Int.Input("tile_width", default=1024, min=64, max=8192, step=32,
                             tooltip="Tile width in pixels, snapped to the 32 px H3 patch grid."),
                io.Int.Input("tile_height", default=1024, min=64, max=8192, step=32,
                             tooltip="Tile height in pixels, snapped to the 32 px H3 patch grid."),
                io.Int.Input("tile_overlap", default=256, min=0, max=8192, step=32,
                             tooltip="Smallest overlap between neighbouring tiles in pixels (32 px grid). Keep it at "
                                     "about a quarter of the tile or more, or the tiles drift apart in content."),
            ],
            outputs=[
                io.Model.Output(display_name="model",
                                tooltip="The patched model: feed it to an H3 sampler or to H3 Split Upscale."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, model, method="trapezoid", tile_width=1024, tile_height=1024, tile_overlap=256) -> io.NodeOutput:
        tw, th, ov = snap_px(tile_width), snap_px(tile_height), snap_px(tile_overlap)
        if min(tw, th) < 2:
            raise ValueError("tile_width and tile_height must be at least 32 px.")
        if ov >= min(tw, th):
            raise ValueError(f"tile_overlap ({ov * 16} px) must be smaller than the tile ({tw * 16} x {th * 16} px) "
                             "so the tiles can advance.")
        patched = model.clone()
        inner = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(H3TiledDiffusion(method, th, tw, ov, inner=inner))
        report = describe_settings(method, tile_width, tile_height, tile_overlap)
        if inner is not None:
            report += f" Runs inside the model function wrapper that was already set ({type(inner).__name__})."
        return io.NodeOutput(patched, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_TiledDiffusion"]
