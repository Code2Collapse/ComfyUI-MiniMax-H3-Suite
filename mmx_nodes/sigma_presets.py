"""Sigma Presets - the sigma schedule, sampler and cfg of a model, as ComfyUI's official templates ship them, with a
graph and a how-to (owner ORDERS 11.6, ledger L8.19). Sources per preset: mmx_utils/sigma_presets.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.sigma_presets import LABELS, PRESETS, how_to, preset_sigmas  # noqa: E402

CATEGORY = "MiniMax H3/Sampling"


def plot(sig: torch.Tensor, audio, title: str, subtitle: str, w: int = 720, h: int = 400) -> torch.Tensor:
    """[1, h, w, 3] graph of sigma per step (and H3's audio clock): drawn with PIL, no extra dependency."""
    from PIL import Image, ImageDraw, ImageFont

    bg, grid, ink, sub = (18, 19, 28), (44, 46, 62), (220, 223, 240), (147, 150, 173)
    cv, ca = (126, 231, 135), (90, 169, 255)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    try:
        f_title, f_small = ImageFont.load_default(size=16), ImageFont.load_default(size=12)
    except TypeError:                                   # Pillow < 10.1
        f_title = f_small = ImageFont.load_default()
    x0, y0, x1, y1 = 58, 54, w - 20, h - 40
    top = max(float(sig.max()), float(audio.max()) if audio is not None else 0.0, 1e-6)
    n = len(sig) - 1
    d.text((x0, 12), title, fill=ink, font=f_title)
    d.text((x0, 32), subtitle, fill=sub, font=f_small)
    for k in range(5):                                  # horizontal grid, labelled in sigma
        y = y1 - (y1 - y0) * k / 4
        d.line([(x0, y), (x1, y)], fill=grid)
        d.text((8, y - 7), f"{top * k / 4:.2f}", fill=sub, font=f_small)
    step = max(1, int(np.ceil(n / 10)))
    for i in range(0, n + 1, step):
        x = x0 + (x1 - x0) * (i / max(1, n))
        d.line([(x, y1), (x, y1 + 4)], fill=sub)
        d.text((x - 4, y1 + 8), str(i), fill=sub, font=f_small)
    d.text((x1 - 34, y1 + 22), "step", fill=sub, font=f_small)

    def curve(vals, col, label, ly):
        pts = [(x0 + (x1 - x0) * (i / max(1, n)), y1 - (y1 - y0) * float(v) / top) for i, v in enumerate(vals)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        for x, y in pts:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
        d.line([(x1 - 150, ly), (x1 - 130, ly)], fill=col, width=3)
        d.text((x1 - 124, ly - 7), label, fill=ink, font=f_small)

    curve(sig.tolist(), cv, "sigma (video)" if audio is not None else "sigma", 16)
    if audio is not None:
        curve(audio.tolist(), ca, "H3 audio clock", 34)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


class MiniMaxH3_SigmaPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SigmaPresets",
            display_name="Sigma Presets (official templates)",
            category=CATEGORY,
            description=(
                "The sigma schedule, sampler and cfg for a model - MiniMax H3 (+ turbo and FastH3), Wan 2.1 / 2.2 / "
                "Animate, Flux.1 / Flux.2, SDXL, SD 1.5, SD 3.5, LTX-2 / 2.3 / 2.5, HunyuanVideo, Qwen-Image, "
                "Z-Image, Chroma, HiDream - as ComfyUI's official workflow templates ship them, with a graph and a "
                "how-to. The curves come from ComfyUI's own scheduler code. A preset that needs a speed LoRA or a "
                "distilled model says so. Connect your model to use its own shift."),
            search_aliases=["sigma presets", "scheduler presets", "sigmas", "steps cfg presets", "sampler settings"],
            inputs=[
                io.Combo.Input("preset", options=LABELS, default=LABELS[0],
                               tooltip="Model and variant. '(template)' = the settings of ComfyUI's official template "
                                       "for that model; the how-to names the template."),
                io.Int.Input("steps", default=0, min=0, max=200,
                             tooltip="0 = the preset's steps. Distilled fixed schedules (LTX stages) ignore it."),
                io.Float.Input("denoise", default=1.0, min=0.01, max=1.0, step=0.01,
                               tooltip="Below 1: the last part of a longer schedule, as BasicScheduler does."),
                io.Int.Input("width", default=1024, min=64, max=8192, step=16,
                             tooltip="Image width: only Flux.2's scheduler depends on it."),
                io.Int.Input("height", default=1024, min=64, max=8192, step=16,
                             tooltip="Image height: only Flux.2's scheduler depends on it."),
                io.Model.Input("model", optional=True,
                               tooltip="Optional: use this model's own sampling (its shift patches included) "
                                       "instead of the preset's."),
            ],
            outputs=[io.Sigmas.Output(display_name="sigmas"), io.Sampler.Output(display_name="sampler"),
                     io.Float.Output(display_name="cfg"), io.Int.Output(display_name="steps"),
                     io.Image.Output(display_name="graph"), io.String.Output(display_name="how_to")],
        )

    @classmethod
    def execute(cls, preset, steps=0, denoise=1.0, width=1024, height=1024, model=None) -> io.NodeOutput:
        import comfy.samplers

        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; pick one from the list")
        p = PRESETS[preset]
        sig, audio, origin = preset_sigmas(preset, steps, denoise, model, width, height)
        text = how_to(preset, sig, audio, origin, steps)
        n = len(sig) - 1
        img = plot(sig, audio, p.label, f"{n} steps, {p.sampler}, {p.guidance}")
        return io.NodeOutput(sig, comfy.samplers.sampler_object(p.sampler), float(p.cfg), n, img, text,
                             ui=ui.PreviewImage(img, cls=cls))


__all__ = ["MiniMaxH3_SigmaPresets", "plot"]
