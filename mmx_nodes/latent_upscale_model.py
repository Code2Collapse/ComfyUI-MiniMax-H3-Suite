# PORTED FROM: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale@0a8d5ee nodes/nodes.py
# (MMH3LatentUpscaleWithModelParams, MMH3LatentUpscaleParams, upscale_latent) and nodes/h3_latent_upscaler.py
# (upscale_video), MIT, Copyright (c) 2026 bbaudio-2025. Upstream bundles the settings for its Ultimate Upscale; here
# it is one node that upscales the latent itself, so it feeds H3 Split Upscale, H3 Tiled Diffusion or any sampler.
"""H3 Latent Upscale (Model) - resize the video half of an H3 AV latent; the audio half passes through untouched."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes.av_noise_mask import _streams, existing_masks  # noqa: E402
from mmx_utils import h3_latent_upscaler as U  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.latent_upscale_model")
CATEGORY = "MiniMax H3/Sampling"
_CACHE: dict = {}


def _model_names() -> list[str]:
    try:
        import folder_paths
        return U.list_models(folder_paths.get_filename_list(U.FOLDER))
    except Exception:  # noqa: BLE001
        return [U.NO_MODEL]


def _nested(parts):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def _device(choice: str):
    if choice == "cpu":
        return torch.device("cpu")
    import comfy.model_management as mm
    return mm.get_torch_device()


def load(model_name: str, precision: str, device):
    import folder_paths
    if str(model_name).startswith("("):
        raise U.UpscalerError(f"No H3 latent upscaler found. Put minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors "
                              f"(or _bf16) from {U.WEIGHTS_URL} into models/latent_upscale_models and restart ComfyUI, "
                              "or pick an interpolation method.")
    path = folder_paths.get_full_path(U.FOLDER, model_name)
    if path is None or not os.path.isfile(path):
        raise U.UpscalerError(f"Model file not found: {model_name} (models/latent_upscale_models).")
    key = (path, os.path.getmtime(path), precision)
    model = _CACHE.get(key)
    if model is None:
        _CACHE.clear()
        model = U.build_model(U.load_state_dict(path), os.path.basename(path)).to(U.DTYPES[precision])
        _CACHE[key] = model
        LOG.info("[H3 Latent Upscale] loaded %s (%s)", os.path.basename(path), precision)
    return model.to(device)


def upscale(latent, method="model", model_name=U.NO_MODEL, width=1280, height=704, device="auto", precision="fp16",
            chunk_latent_steps=0, chunk_overlap_steps=4, offload_model=True):
    video, audio = _streams(latent, "H3 Latent Upscale")
    if method not in U.METHODS:
        raise U.UpscalerError(f"method must be one of {', '.join(U.METHODS)}")
    h_out, w_out = U.target_cells(width, height)
    _, c, t, h, w = video.shape
    if c != 24:
        raise U.UpscalerError(f"expected the 24-channel H3 video latent, got {c} channels")
    lines = [f"video latent {h} x {w} cells ({h * 16} x {w * 16} px), {t} latent steps -> {h_out} x {w_out} cells "
             f"({h_out * 16} x {w_out * 16} px), scale {w_out / w:.3f} x {h_out / h:.3f}; audio untouched"]
    if (h_out, w_out) == (h, w):
        lines.append("target equals the input size: unchanged")
        return dict(latent), "\n".join(lines)
    if method == "model":
        if h_out < h or w_out < w:
            raise U.UpscalerError("The H3 latent upscaler only enlarges (trained for scales 1-4). Use an "
                                  "interpolation method to shrink.")
        dev = _device(device)
        dtype = torch.float32 if dev.type == "cpu" else U.DTYPES[precision]
        if dev.type != "cpu":
            try:     # weights (~0.7 GB fp16) + a few 512-channel activations at the output size
                import comfy.model_management as mm
                mm.free_memory(int(512 * t * h_out * w_out * 2 * 6 + 0.7e9), dev)
            except Exception:  # noqa: BLE001
                pass
        model = load(model_name, "fp32" if dtype == torch.float32 else precision, dev)
        try:
            up = U.run_model(model, video, h_out, w_out, dev, dtype, int(chunk_latent_steps), int(chunk_overlap_steps))
        finally:
            if offload_model:
                model.to("cpu")
                try:
                    import comfy.model_management as mm
                    mm.soft_empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        ranges = U.chunk_ranges(t, int(chunk_latent_steps), int(chunk_overlap_steps))
        lines.append(f"H3 3D latent upscaler {model_name} on {dev} ({str(dtype).replace('torch.', '')}); "
                     + ("whole clip in one pass" if len(ranges) == 1 else
                        f"{len(ranges)} temporal chunks, cross-faded (approximate: the network's GroupNorm and 3x3x3 "
                        "convolutions see the whole clip)"))
    else:
        up = U.interpolate_video(video, h_out, w_out, method)
        lines.append(f"{method} interpolation, no model")
    out = dict(latent)
    out["samples"] = _nested((up.to(video.dtype), audio))
    vm, am = existing_masks(latent)
    if vm is not None:
        vm5 = vm if vm.ndim == 5 else vm.reshape((1, 1) + tuple(vm.shape[-3:]))
        vm5 = U.interpolate_video(vm5.float(), h_out, w_out, "nearest-exact")
        out["noise_mask"] = _nested((vm5, am if am is not None else torch.ones_like(audio[:, :1])))
        lines.append("the video noise mask was resized with it (nearest)")
    return out, "\n".join(lines)


SIZE_BY = ("width / height", "scale", "megapixels")
PX_PER_CELL = 16          # H3's VAE: one latent cell per 16 x 16 pixels


def target_size(latent, size_by, width, height, scale=2.0, megapixels=1.0):
    """(width, height, note) in pixels for the chosen sizing; upscale() snaps them to the 32 px grid."""
    if size_by == SIZE_BY[0]:
        return int(width), int(height), ""
    video, _audio = _streams(latent, "H3 Latent Upscale")
    cur_w, cur_h = int(video.shape[-1]) * PX_PER_CELL, int(video.shape[-2]) * PX_PER_CELL
    if size_by == "scale":
        k = float(scale)
        how = f"size: {cur_w}x{cur_h} x {k:g}"
    elif size_by == "megapixels":
        k = (float(megapixels) * 1e6 / float(cur_w * cur_h)) ** 0.5
        how = f"size: {cur_w}x{cur_h} at {float(megapixels):g} MP (x {k:.3f})"
    else:
        raise ValueError(f"H3 Latent Upscale: size_by must be one of {', '.join(SIZE_BY)}; got {size_by!r}")
    w, h = max(32, round(cur_w * k)), max(32, round(cur_h * k))
    return w, h, f"{how} -> {w}x{h} before the 32 px grid"


class MiniMaxH3_LatentUpscaleModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_LatentUpscaleModel",
            display_name="H3 Latent Upscale (Model)",
            category=CATEGORY,
            description=(
                "Resize the video half of an H3 AV latent to a new pixel size, with the trained H3 3D latent upscaler "
                "(LBH-123-AI, models/latent_upscale_models, .safetensors only) or plain interpolation. The audio half "
                "passes through untouched. Then re-sample it (H3 Split Upscale, H3 Tiled Diffusion or a sampler at "
                "low denoise) to add detail. The conditioning must be made for the new size."),
            search_aliases=["h3 upscale", "latent upscale model", "h3 latent upscaler", "upscale latent"],
            inputs=[
                io.Latent.Input("latent", tooltip="An H3 video + audio latent."),
                io.Combo.Input("method", options=list(U.METHODS), default="model",
                               tooltip="model: the trained 3D upscaler (sharper, needs the checkpoint). The others "
                                       "interpolate the latent without a model (as core's Upscale Latent)."),
                io.Combo.Input("model_name", options=_model_names(),
                               tooltip=f"Checkpoint in models/latent_upscale_models. Download "
                                       f"minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors from {U.WEIGHTS_URL}. "
                                       "Pickle files (.pth) are refused."),
                io.Int.Input("width", default=1280, min=64, max=8192, step=32,
                             tooltip="Target width in pixels (32 px grid)."),
                io.Int.Input("height", default=704, min=64, max=8192, step=32,
                             tooltip="Target height in pixels (32 px grid)."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",
                               tooltip="auto: ComfyUI's device (GPU if there is one)."),
                io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16",
                               tooltip="Compute precision on the GPU. On the CPU it always runs fp32."),
                io.Int.Input("chunk_latent_steps", default=0, min=0, max=4096, advanced=True,
                             tooltip="0: the whole clip in one pass (exact). > 0: chunks of this many latent steps "
                                     "(4 pixel frames each), cross-faded; less memory, approximate."),
                io.Int.Input("chunk_overlap_steps", default=4, min=0, max=64, advanced=True,
                             tooltip="Latent steps shared by neighbouring chunks, cross-faded."),
                io.Boolean.Input("offload_model", default=True, advanced=True,
                                 tooltip="Move the upscaler back to the CPU afterwards, so the H3 model and the "
                                         "upscaler are never both on the GPU."),
                # C2C L9.39, appended (saved widgets keep their positions): size by factor or pixel budget as the
                # reference upscalers do, not only by width / height
                io.Combo.Input("size_by", options=list(SIZE_BY), default=SIZE_BY[0], optional=True,
                               tooltip="width / height: the two size widgets. scale: the latent's own size times "
                                       "'scale'. megapixels: the latent's aspect at about this many megapixels. All "
                                       "snap to H3's 32 px grid."),
                io.Float.Input("scale", default=2.0, min=0.25, max=8.0, step=0.05, optional=True,
                               tooltip="size_by = scale: multiply the current size by this."),
                io.Float.Input("megapixels", default=1.0, min=0.05, max=33.0, step=0.05, optional=True,
                               tooltip="size_by = megapixels: the target pixel count in millions, aspect kept."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, latent, method="model", model_name=U.NO_MODEL, width=1280, height=704, device="auto",
                precision="fp16", chunk_latent_steps=0, chunk_overlap_steps=4, offload_model=True,
                size_by=None, scale=2.0, megapixels=1.0) -> io.NodeOutput:
        width, height, how = target_size(latent, size_by or SIZE_BY[0], width, height, scale, megapixels)
        out, report = upscale(latent, method, model_name, width, height, device, precision, chunk_latent_steps,
                              chunk_overlap_steps, offload_model)
        if how:
            report = f"{how}\n{report}"
        return io.NodeOutput(out, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_LatentUpscaleModel", "upscale"]
