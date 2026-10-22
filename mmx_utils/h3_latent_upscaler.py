# PORTED FROM: https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale@0a8d5ee nodes/h3_latent_upscaler.py
# (MIT, Copyright (c) 2026 bbaudio-2025), itself a copy of the inference code of
# https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (MIT). Checkpoints:
# huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0), minimax_h3_latent_upscaler_3d_conv_v1_*.safetensors.
# Kept from upstream: the network, the architecture detection, safetensors only (pickle refused before the file is
# read, CVE-2025-32434), the 24-channel check. Changed here: normalisation constants from the weights repo's
# config.json, no folder registration (core 0.39 registers latent_upscale_models), optional temporal chunks with a
# cross-fade, interpolation and the target-size maths in one place, plain-English errors.
"""H3 3D latent upscaler: resize the video half of an H3 AV latent with a trained 3D network (or interpolation)."""

from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_DOWNSAMPLE = 16
FOLDER = "latent_upscale_models"
METHODS = ("model", "bilinear", "bicubic", "area", "nearest-exact")
SUPPORTED_EXT = (".safetensors", ".sft")
REJECTED_EXT = (".pth", ".pt", ".pt2", ".ckpt", ".bin", ".pkl")
NO_MODEL = "(no .safetensors H3 latent upscaler in models/latent_upscale_models)"
WEIGHTS_URL = "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"

# per-channel mean / std of the H3 VAE latent (config.json "latent_normalization" of the weights repo)
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class UpscalerError(ValueError):
    pass


def _norm(channels):
    return nn.GroupNorm(32, channels)


def _zero(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _norm(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, t, hh, w = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2).transpose(1, 2)
        v = self.v(h).flatten(2).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj_out(h.transpose(1, 2).reshape(b, c, t, hh, w))


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0.0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(_norm(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = _norm(self.out_channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(p=dropout),
                                        _zero(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        e = self.emb_layers(emb).type(h.dtype)
        while e.ndim < h.ndim:
            e = e[..., None]
        scale, shift = torch.chunk(e, 2, dim=1)
        h = self.out_layers(self.out_norm(h) * (1 + scale) + shift)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                                padding=(kernel_size // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        def stage(n):
            blocks = nn.ModuleList()
            for b in range(n):
                if (b == 1 or b == n - 1) and attn:
                    blocks.append(AttnBlock3D(channels))
                blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
                if temporal_every > 0 and b % temporal_every == 0:
                    blocks.append(TemporalConv(channels, temporal_kernel))
            return blocks

        self.in_blocks = stage(in_blocks)
        self.out_blocks = stage(out_blocks)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = tuple(target_size)
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x
        if size == tuple(x.shape[-3:]):
            return x
        emb = self.embed(torch.tensor([scale - 1 if scale is not None else 0.0], dtype=x.dtype,
                                      device=x.device).unsqueeze(0))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


def model_ext(name: str) -> str:
    return os.path.splitext(str(name))[1].lower()


def list_models(names) -> list[str]:
    usable = sorted(n for n in names if model_ext(n) in SUPPORTED_EXT)
    return usable or [NO_MODEL]


def load_state_dict(path: str) -> dict:
    """safetensors only, refused before the file is opened otherwise."""
    if model_ext(path) not in SUPPORTED_EXT:
        raise UpscalerError(
            f"Refusing to load '{os.path.basename(path)}': only {' / '.join(SUPPORTED_EXT)} checkpoints are accepted. "
            "Pickle formats (.pth / .pt / .ckpt) can run arbitrary code when loaded. Use the fp16 or bf16 "
            f".safetensors file from {WEIGHTS_URL}.")
    from safetensors.torch import load_file
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return {k: v.to(torch.float16) if v.dtype == getattr(torch, "float8_e4m3fn", None) else v for k, v in sd.items()}


def detect_arch(sd: dict) -> dict:
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512, "dropout": 0.1, "attn": False,
           "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = int(sd["conv_in.weight"].shape[1])
        cfg["channels"] = int(sd["conv_in.weight"].shape[0])
    in_ids, out_ids, temporal = set(), set(), False
    for k in sd:
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        if k.endswith("dwconv.weight"):
            temporal = True
            cfg["temporal_kernel"] = int(sd[k].shape[2])
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    cfg["temporal_every"] = 2 if temporal else 0
    cfg["attn"] = any(k.split(".")[2:3] == ["q"] for k in sd if k.startswith(("in_blocks.", "out_blocks.")))
    return cfg


def build_model(sd: dict, name: str = "checkpoint") -> LatentResizer3D:
    cfg = detect_arch(sd)
    if cfg["in_channels"] != 24:
        raise UpscalerError(f"'{name}' is not an H3 latent upscaler: it takes {cfg['in_channels']} channels, the H3 "
                            "video latent has 24.")
    model = LatentResizer3D(**cfg)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise UpscalerError(f"'{name}' does not match the H3 3D latent upscaler architecture: "
                            f"{str(exc).splitlines()[0]}") from exc
    return model.eval().requires_grad_(False)


def target_cells(width: int, height: int) -> tuple[int, int]:
    """Pixel size -> (h, w) latent cells, snapped to the 32 px patch grid."""
    w = max(2, int(round(int(width) / 32.0)) * 2)
    h = max(2, int(round(int(height) / 32.0)) * 2)
    return h, w


def interpolate_video(video: torch.Tensor, h_out: int, w_out: int, method: str) -> torch.Tensor:
    b, c, t, h, w = video.shape
    if (h, w) == (h_out, w_out):
        return video
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kw = {"align_corners": False} if method in ("bilinear", "bicubic") else {}
    up = F.interpolate(flat.float(), size=(h_out, w_out), mode=method, **kw).to(video.dtype)
    return up.reshape(b, t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()


def chunk_ranges(t: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """[(start, end)] over [0, t) latent steps; chunk <= 0 or >= t is one range."""
    t, chunk, overlap = int(t), int(chunk), int(overlap)
    if chunk <= 0 or chunk >= t:
        return [(0, t)]
    overlap = max(0, min(overlap, chunk - 1))
    out, s = [], 0
    while True:
        e = min(s + chunk, t)
        out.append((s, e))
        if e >= t:
            return out
        s = e - overlap


def run_model(model: nn.Module, video: torch.Tensor, h_out: int, w_out: int, device, dtype,
              chunk: int = 0, overlap: int = 4) -> torch.Tensor:
    """Upscale [B, 24, T, h, w] with the network; chunks along T are cross-faded over their overlap."""
    b, c, t, h, w = video.shape
    scale = ((w_out / w) + (h_out / h)) / 2.0
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    out = torch.zeros((b, c, t, h_out, w_out), dtype=torch.float32)
    weight = torch.zeros((1, 1, t, 1, 1), dtype=torch.float32)
    ranges = chunk_ranges(t, chunk, overlap)
    for i, (s, e) in enumerate(ranges):
        x = video[:, :, s:e].to(device=device, dtype=dtype)
        with torch.inference_mode():
            y = model((x - mean) / std, scale=scale, target_size=(e - s, h_out, w_out)) * std + mean
        y = y.float().cpu()
        wt = torch.ones(e - s, dtype=torch.float32)
        if i > 0:                                   # ramp in over the overlap with the previous chunk
            ov = ranges[i - 1][1] - s
            if ov > 0:
                wt[:ov] = torch.linspace(0.0, 1.0, ov + 2)[1:-1]
        if i + 1 < len(ranges):                     # ramp out over the overlap with the next chunk
            ov = e - ranges[i + 1][0]
            if ov > 0:
                wt[e - s - ov:] = torch.minimum(wt[e - s - ov:], torch.linspace(1.0, 0.0, ov + 2)[1:-1])
        out[:, :, s:e] += y * wt.view(1, 1, -1, 1, 1)
        weight[:, :, s:e] += wt.view(1, 1, -1, 1, 1)
    return (out / weight.clamp(min=1e-12)).to(video.dtype)
