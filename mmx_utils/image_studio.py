# SPDX-License-Identifier: Unlicense
# PORTED FROM: ComfyUI-MiniMax-H3-Image-Studio :: nodes.py @ upstream
"""Pure logic for H3 still-image generation — no ComfyUI imports."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .h3_constants import CANVAS_MULTIPLE, MAX_PIXELS
from .h3_grid import decoded_frames_for_latent_t, latent_t_for_frame_count

MEBIPIXEL = 1024 * 1024
NATIVE_MAX_PIXELS = MAX_PIXELS
REF_IMAGE_SHORT_EDGE = 2048
MAX_REFERENCE_IMAGES = 9

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 square": (1, 1),
    "4:5 portrait": (4, 5),
    "3:4 portrait": (3, 4),
    "2:3 portrait": (2, 3),
    "9:16 portrait": (9, 16),
    "16:9 landscape": (16, 9),
    "3:2 landscape": (3, 2),
    "4:3 landscape": (4, 3),
    "21:9 ultrawide": (21, 9),
}

SINGLE_IMAGE_FRAME_PROFILE = "single image | 1 frame (image VAE)"
RECOMMENDED_FRAME_PROFILE = "recommended | 5 frames"
BALANCED_FRAME_PROFILE = "extended quality | 9 frames"
HIGH_FRAME_PROFILE = "high quality | 13 frames"
MAX_QUALITY_FRAME_PROFILE = "maximum quality | 20 frames (slow)"

FRAME_PRESETS: Dict[str, int] = {
    SINGLE_IMAGE_FRAME_PROFILE: 1,
    RECOMMENDED_FRAME_PROFILE: 5,
    BALANCED_FRAME_PROFILE: 9,
    HIGH_FRAME_PROFILE: 13,
    MAX_QUALITY_FRAME_PROFILE: 20,
}

RESOLUTION_PROFILES: Dict[str, Optional[float]] = {
    "fast preview | 0.40 MP": 0.40,
    "balanced | 0.70 MP": 0.70,
    "high | 0.90 MP": 0.90,
    "native detail | 0.98 MP": 0.98,
    "high-res | 2.00 MP": 2.00,
    "ultra | 4.00 MP": 4.00,
    "ultra+ | 8.00 MP": 8.00,
    "custom megapixels": None,
}

SAMPLING_PROFILES: Dict[str, Tuple[str, str, int, float, float]] = {
    "base quality | RES 20 steps": ("res_multistep", "simple", 20, 12.0, 3.0),
    "base speed | RES 12 steps": ("res_multistep", "simple", 12, 12.0, 3.0),
    "Turbo v1.0 | 8 steps": ("euler", "simple", 8, 12.0, 3.0),
    "Turbo v1.0 768p | 4 steps": ("euler", "simple", 4, 6.0, 3.0),
    "FL2VA Turbo v1.0 768p | 8 steps": ("euler", "simple", 8, 6.0, 3.0),
    "FL2VA Turbo v1.2 768p | 4 steps": ("euler", "simple", 4, 6.0, 3.0),
    "REF2VA Turbo v1.0 768p | 8 steps": ("euler", "simple", 8, 12.0, 3.0),
    "REF2VA Turbo v0.1 | 4 steps": ("euler", "simple", 4, 12.0, 3.0),
    "hybrid single image | ER-SDE 8 steps": ("er_sde", "sgm_uniform", 8, 12.0, 3.0),
}

CUSTOM_SAMPLING_PROFILE = "custom | use controls below"

LEGACY_SAMPLING_PROFILES: Dict[str, Tuple[str, str, int, float, float]] = {
    "quality | 20 steps": SAMPLING_PROFILES["base quality | RES 20 steps"],
    "speed | 12 steps": SAMPLING_PROFILES["base speed | RES 12 steps"],
    "LightX v0.1 | ER-SDE 4 steps": ("er_sde", "simple", 4, 12.0, 3.0),
    "LightX v0.1 | SA-Solver 4 steps": ("sa_solver", "simple", 4, 12.0, 3.0),
    "turbo | 8 steps (LoRA)": ("res_multistep", "simple", 8, 12.0, 4.0),
    "turbo | 4 steps (LoRA, experimental)": ("res_multistep", "simple", 4, 12.0, 4.0),
}

VIDEO_PROMPT_RE = re.compile(
    r"(?:\b(?:video|animation|timeline|storyboard|fps|seconds?)\b|"
    r"\[(?:\d+(?:\.\d+)?s?\s*[-–]\s*)?\d+(?:\.\d+)?s\]|"
    r"\b(?:camera movement|push[- ]?in|zoom|pan|dolly|cut to|hard cuts?)\b|"
    r"\b(?:overall_soundscape|non_diegetic_music|audio|soundtrack)\s*:)",
    re.IGNORECASE,
)


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def fit_area_to_ratio(
    area: float,
    ratio: float,
    multiple: int,
    cap_pixels: Optional[int],
) -> Tuple[int, int]:
    """Fit an area while preserving aspect ratio on H3's resolution grid."""
    ratio = max(1e-6, float(ratio))
    area = max(float(multiple * multiple), float(area))
    target_area = min(area, float(cap_pixels)) if cap_pixels is not None else area
    ideal_w = math.sqrt(target_area * ratio)
    ideal_h = math.sqrt(target_area / ratio)
    width = round_to_multiple(ideal_w, multiple)
    height = round_to_multiple(ideal_h, multiple)

    if cap_pixels is None or width * height <= cap_pixels:
        return width, height

    center_w = max(1, int(round(ideal_w / multiple)))
    center_h = max(1, int(round(ideal_h / multiple)))
    candidates = []
    for wi in range(max(1, center_w - 6), center_w + 7):
        for hi in range(max(1, center_h - 6), center_h + 7):
            w = wi * multiple
            h = hi * multiple
            pixels = w * h
            if pixels > cap_pixels:
                continue
            aspect_error = abs(math.log((w / h) / ratio))
            area_error = abs(pixels - target_area) / max(1.0, target_area)
            candidates.append((3.0 * aspect_error + area_error, -pixels, w, h))

    if not candidates:
        max_cells = max(1, int(target_area // (multiple * multiple)))
        fallback = []
        for hi in range(1, max_cells + 1):
            ideal_wi = ratio * hi
            max_wi = max(1, max_cells // hi)
            width_cells = {
                max(1, min(max_wi, int(math.floor(ideal_wi)))),
                max(1, min(max_wi, int(round(ideal_wi)))),
                max(1, min(max_wi, int(math.ceil(ideal_wi)))),
                max_wi,
            }
            for wi in width_cells:
                cells = wi * hi
                if cells > max_cells:
                    continue
                w = wi * multiple
                h = hi * multiple
                pixels = w * h
                aspect_error = abs(math.log((w / h) / ratio))
                area_error = abs(pixels - target_area) / max(1.0, target_area)
                fallback.append((3.0 * aspect_error + area_error, -pixels, w, h))
        if not fallback:
            return multiple, multiple
        _, _, width, height = min(fallback)
        return width, height
    _, _, width, height = min(candidates)
    return width, height


def resolve_frame_count(frame_preset: str) -> int:
    if frame_preset in FRAME_PRESETS:
        return FRAME_PRESETS[frame_preset]
    raise ValueError(f"Unknown H3 image quality profile: {frame_preset}")


def frame_profile_packet(frame_preset: str) -> tuple[int, int, int]:
    """Return (requested_frames, latent_t, natural_decoded_frames) for a profile."""
    requested = resolve_frame_count(frame_preset)
    latent_t, natural = latent_t_for_frame_count(requested)
    return requested, latent_t, natural


def resize_image(image: torch.Tensor, width: int, height: int, fit_mode: str) -> torch.Tensor:
    """Resize Comfy IMAGE [B,H,W,C] to [B,height,width,3] without mutating input."""
    image = image[..., :3].clone()
    samples = image.movedim(-1, 1)

    if fit_mode == "stretch":
        out = F.interpolate(samples, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
        return out.movedim(1, -1)

    if fit_mode == "crop_center":
        src_h, src_w = int(samples.shape[-2]), int(samples.shape[-1])
        scale = max(width / src_w, height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = F.interpolate(samples, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        y0 = max(0, (new_h - height) // 2)
        x0 = max(0, (new_w - width) // 2)
        cropped = resized[:, :, y0:y0 + height, x0:x0 + width]
        return cropped.movedim(1, -1)

    # contain_pad
    src_h, src_w = int(samples.shape[-2]), int(samples.shape[-1])
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = F.interpolate(samples, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    pad_l = (width - new_w) // 2
    pad_r = width - new_w - pad_l
    pad_t = (height - new_h) // 2
    pad_b = height - new_h - pad_t
    padded = F.pad(resized, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    return padded.movedim(1, -1).clamp(0.0, 1.0)


def normalize_prompt(
    mode: str,
    prompt: str,
    optimize_prompt: bool,
    preserve_strength: float,
    reference_count: int = 1,
) -> str:
    prompt = (prompt or "").strip()
    if not optimize_prompt:
        return prompt

    preserve_strength = float(max(0.0, min(1.0, preserve_strength)))
    if preserve_strength >= 0.8:
        preserve = (
            "Preserve unmentioned source traits. Requested changes take priority over "
            "identity, pose, composition, and geometry preservation."
        )
    elif preserve_strength >= 0.5:
        preserve = (
            "Preserve identity, pose, composition, perspective, and object geometry "
            "unless the edit requires a change."
        )
    else:
        preserve = "Keep the source recognizable while applying the requested changes."

    still = (
        "Create one finished still image, not a collage, split screen, sequence, or "
        "duplicated subject. Do not add motion, cuts, or audio."
    )

    if mode == "text_to_image (FL2VA)":
        return f"{still}\n\nTarget image description: {prompt}"
    if mode == "image_to_image (FL2VA)":
        return (
            f"<Picture 1> is the source image. Apply the requested edit. {still} {preserve}\n\n"
            f"Target edit: {prompt}"
        )
    if preserve_strength >= 0.8:
        primary_rule = "Preserve only the unmentioned identity and appearance traits of <Picture 1>."
    elif preserve_strength >= 0.5:
        primary_rule = "Keep the unmentioned identity traits of <Picture 1> recognizable."
    else:
        primary_rule = (
            "Use unmentioned traits from <Picture 1> only when they do not conflict "
            "with the requested result."
        )
    additional_tags = ", ".join(f"<Picture {index}>" for index in range(2, reference_count + 1))
    if additional_tags:
        additional_rule = (
            f" The connected references are <Picture 1> through <Picture {reference_count}>. "
            "The target instructions take priority over source preservation. When a trait is assigned to a picture, "
            "copy that trait from that picture even when it conflicts with <Picture 1>; do not revert the requested "
            "change to match <Picture 1>. Use each picture only for its explicitly assigned traits."
        )
    else:
        additional_rule = (
            " <Picture 1> is the source reference. The target instructions take priority over source preservation."
        )
    return (
        f"Follow the target image instructions exactly.{additional_rule} {primary_rule} "
        f"{still}\n\nTarget image instructions: {prompt}"
    )


def prompt_warning(prompt: str) -> str:
    if VIDEO_PROMPT_RE.search(prompt or ""):
        return (
            " WARNING: prompt contains motion, timeline, camera, or audio terms. "
            "Describe one final still image instead."
        )
    return ""


def reference_resize(
    image: torch.Tensor,
    generation_width: int,
    generation_height: int,
    reference_size: str,
) -> Tuple[torch.Tensor, int, int]:
    image = image[:1, ..., :3].clone()
    h, w = int(image.shape[1]), int(image.shape[2])
    if reference_size == "max_identity_2048":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((generation_width * generation_height) / max(1, w * h)))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = resize_image(image, tw, th, "stretch")
    return resized, tw, th


def collect_reference_images(
    source_image: torch.Tensor,
    additional_images: Sequence[Optional[torch.Tensor]],
) -> List[torch.Tensor]:
    references: List[torch.Tensor] = []
    for image in (source_image, *additional_images):
        if image is None:
            continue
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("Every REF2VA reference must be a non-empty ComfyUI IMAGE batch [B,H,W,C].")
        references.append(image[:1].clone())
        if len(references) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"MiniMax H3 REF2VA supports at most {MAX_REFERENCE_IMAGES} reference images."
            )
    return references


def resolve_source_ratio(
    aspect_ratio: str,
    source_image: Optional[torch.Tensor],
) -> Tuple[float, str]:
    if aspect_ratio == "source image":
        if source_image is None:
            raise ValueError(
                'Aspect ratio "source image" requires a connected source_image. '
                "Connect an IMAGE or choose an explicit aspect ratio."
            )
        if not isinstance(source_image, torch.Tensor) or source_image.ndim != 4 or source_image.shape[0] < 1:
            raise ValueError("source_image must be a non-empty ComfyUI IMAGE batch [B,H,W,C].")
        h, w = int(source_image.shape[1]), int(source_image.shape[2])
        if h < 1 or w < 1:
            raise ValueError("source_image has invalid spatial dimensions.")
        return w / h, f"source ratio {w}:{h}"

    rw, rh = ASPECT_RATIOS[aspect_ratio]
    return rw / rh, aspect_ratio


def calculate_custom_resolution(
    aspect_ratio: str,
    megapixels: float,
    multiple: int,
    native_area_cap: bool,
    custom_width: int,
    custom_height: int,
    source_image: Optional[torch.Tensor] = None,
) -> Tuple[int, int, str]:
    multiple = int(multiple)
    cap = NATIVE_MAX_PIXELS if native_area_cap else None

    if aspect_ratio == "custom dimensions":
        width = round_to_multiple(custom_width, multiple)
        height = round_to_multiple(custom_height, multiple)
        if cap is not None and width * height > cap:
            width, height = fit_area_to_ratio(cap, width / height, multiple, cap)
        source = "custom"
    else:
        ratio, source = resolve_source_ratio(aspect_ratio, source_image)
        target_area = float(megapixels) * MEBIPIXEL
        if cap is not None:
            target_area = min(target_area, cap)
        width, height = fit_area_to_ratio(target_area, ratio, multiple, cap)

    mp = width * height / MEBIPIXEL
    cap_text = "native cap on" if native_area_cap else "oversize experimental"
    oversize_note = (
        " | WARNING: H3-Base is a 768p model; direct oversize does not reproduce the unreleased H3-Regenerate-2K pipeline"
        if not native_area_cap and width * height > NATIVE_MAX_PIXELS
        else ""
    )
    report = (
        f"{width}×{height} | {mp:.3f} MP (1024²) | {source} | multiple {multiple} | "
        f"{cap_text}{oversize_note}"
    )
    return width, height, report


def calculate_preset_resolution(
    aspect_ratio: str,
    resolution_profile: str,
    source_image: Optional[torch.Tensor] = None,
    custom_megapixels: float = 2.0,
    limit_to_native_area: bool = False,
) -> Tuple[int, int, str]:
    ratio, source = resolve_source_ratio(aspect_ratio, source_image)

    target_mp = RESOLUTION_PROFILES[resolution_profile]
    if target_mp is None:
        target_mp = max(0.10, min(64.0, float(custom_megapixels)))
    cap = NATIVE_MAX_PIXELS if limit_to_native_area else None
    width, height = fit_area_to_ratio(target_mp * MEBIPIXEL, ratio, CANVAS_MULTIPLE, cap)
    actual_mp = width * height / MEBIPIXEL
    native_scale = width * height / NATIVE_MAX_PIXELS
    if limit_to_native_area:
        size_note = "native area limiter on"
    elif width * height > NATIVE_MAX_PIXELS:
        size_note = (
            f"UNLOCKED oversize (~{native_scale:.1f}× native pixel area; VRAM and attention cost rise sharply; "
            "detail gain is checkpoint-dependent)"
        )
    else:
        size_note = "within native area"
    profile_note = (
        f"custom {target_mp:.2f} MP" if resolution_profile == "custom megapixels" else resolution_profile
    )
    report = (
        f"{width}×{height} | {actual_mp:.3f} MP (1024²) | {source} | "
        f"{profile_note} | {size_note}"
    )
    return width, height, report


def first_stable_edit_frame(images: torch.Tensor, max_side: int = 256) -> Tuple[int, float]:
    if images.ndim != 4 or images.shape[0] <= 1:
        return 0, 0.0

    x = images[..., :3].movedim(-1, 1).float().clone()
    height, width = x.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        x = F.interpolate(
            x,
            size=(max(16, round(height * scale)), max(16, round(width * scale))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    change = (x - x[:1]).abs().mean(dim=(1, 2, 3))
    robust_peak = torch.quantile(change[1:], 0.90)
    mature = change >= robust_peak * 0.80

    for index in range(1, len(change) - 1):
        if bool(mature[index]) and bool(mature[index + 1]):
            return index, float(change[index].item())
    index = int(torch.argmax(change[1:]).item()) + 1
    return index, float(change[index].item())


def stable_quality_frame(images: torch.Tensor, max_side: int = 256) -> Tuple[int, float]:
    if images.ndim != 4 or images.shape[0] <= 1:
        return 0, 1.0

    x = images[..., :3].movedim(-1, 1).float().clone()
    height, width = x.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        x = F.interpolate(
            x,
            size=(max(16, round(height * scale)), max(16, round(width * scale))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    x = x.clamp(0.0, 1.0)

    def minmax(values: torch.Tensor) -> torch.Tensor:
        spread = values.max() - values.min()
        if float(spread.abs()) < 1e-8:
            return torch.ones_like(values)
        return (values - values.min()) / spread

    gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    sharpness = torch.log1p(F.conv2d(gray, lap_kernel, padding=1).var(dim=(1, 2, 3)) * 1000.0)
    contrast = gray.std(dim=(1, 2, 3))
    clipped = ((x < 0.01) | (x > 0.99)).float().mean(dim=(1, 2, 3))
    exposure = (1.0 - clipped * 3.0).clamp(0.0, 1.0)
    quality = 0.70 * minmax(sharpness) + 0.20 * minmax(contrast) + 0.10 * exposure

    temporal_delta = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    temporal_delta[0] = (x[0] - x[1]).abs().mean()
    temporal_delta[-1] = (x[-1] - x[-2]).abs().mean()
    if x.shape[0] > 2:
        temporal_delta[1:-1] = 0.5 * (x[1:-1] - x[:-2]).abs().mean(dim=(1, 2, 3))
        temporal_delta[1:-1] += 0.5 * (x[1:-1] - x[2:]).abs().mean(dim=(1, 2, 3))
    scores = 0.80 * quality + 0.20 * (1.0 - minmax(temporal_delta))
    index = int(torch.argmax(scores).item())
    return index, float(scores[index].item())


def _metric_tensor(frames: torch.Tensor, max_side: int = 512, chunk_size: int = 4) -> torch.Tensor:
    samples = frames[..., :3].movedim(-1, 1)
    h, w = samples.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return samples.float().clamp(0.0, 1.0)

    nh = max(16, int(round(h * scale)))
    nw = max(16, int(round(w * scale)))
    chunks = []
    for chunk in samples.split(max(1, int(chunk_size)), dim=0):
        chunk = chunk.float()
        chunk = F.interpolate(
            chunk,
            size=(nh, nw),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        chunks.append(chunk)
    return torch.cat(chunks, dim=0).clamp(0.0, 1.0)


def _minmax(values: torch.Tensor) -> torch.Tensor:
    lo = values.min()
    hi = values.max()
    if float((hi - lo).abs()) < 1e-8:
        return torch.ones_like(values)
    return (values - lo) / (hi - lo)


def _quality_metrics(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    sharpness = torch.log1p(lap.var(dim=(1, 2, 3)) * 1000.0)
    contrast = gray.std(dim=(1, 2, 3))
    clipped = ((x < 0.01) | (x > 0.99)).float().mean(dim=(1, 2, 3))
    exposure = (1.0 - clipped * 3.0).clamp(0.0, 1.0)
    return sharpness, contrast, exposure


def _similarity(x: torch.Tensor, source_image: torch.Tensor) -> torch.Tensor:
    ref = source_image[:1, ..., :3].movedim(-1, 1).to(device=x.device, dtype=x.dtype).clone()
    ref = F.interpolate(ref, size=x.shape[-2:], mode="bilinear", align_corners=False, antialias=True).clamp(0.0, 1.0)
    ref = ref.expand(x.shape[0], -1, -1, -1)
    color_error = (x - ref).abs().mean(dim=(1, 2, 3))

    def gradients(t: torch.Tensor):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy

    gx, gy = gradients(x)
    rgx, rgy = gradients(ref)
    edge_error = 0.5 * (gx - rgx).abs().mean(dim=(1, 2, 3)) + 0.5 * (gy - rgy).abs().mean(dim=(1, 2, 3))
    return (1.0 - (0.75 * color_error + 0.25 * edge_error)).clamp(0.0, 1.0)


def select_still_frame(
    frames: torch.Tensor,
    strategy: str,
    manual_index: int = 0,
    skip_first_frames: int = 0,
    candidate_start: float = 0.0,
    candidate_end: float = 1.0,
    similarity_weight: float = 0.60,
    top_k: int = 4,
    source_image: Optional[torch.Tensor] = None,
    emit_candidate_batch: bool = False,
    recommended_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, str]:
    """Pick one still from a decoded frame batch; returns clones, never views."""
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("frames must be a non-empty ComfyUI IMAGE batch [N,H,W,C]")

    n = int(frames.shape[0])
    fixed_indices = {
        "decode_recommended": 0 if recommended_index is None else int(recommended_index),
        "first": 0,
        "manual_index": int(manual_index),
        "middle": n // 2,
        "last": n - 1,
    }
    if strategy in fixed_indices:
        selected_index = max(0, min(n - 1, fixed_indices[strategy]))
        chosen = frames[selected_index:selected_index + 1].clone()
        primary = frames.clone() if emit_candidate_batch else chosen
        debug = chosen.clone() if emit_candidate_batch else frames.new_empty((0, *frames.shape[1:]))
        report = f"Fixed strategy={strategy}; frame {selected_index}/{n - 1}."
        if strategy == "decode_recommended" and recommended_index is None:
            report += " recommended_index was not connected, so frame 0 was used."
        if emit_candidate_batch:
            report += f" selected_image emits the complete {n}-frame batch; candidate_batch_debug contains the chosen frame."
        return primary, debug, selected_index, 1.0, report

    start = max(int(skip_first_frames), int(math.floor(max(0.0, min(1.0, candidate_start)) * n)))
    start = min(n - 1, start)
    end = int(math.ceil(max(0.0, min(1.0, candidate_end)) * n))
    end = max(start + 1, min(n, end))
    candidate_indices = torch.arange(start, end, device=frames.device)
    candidate_frames = frames[start:end]
    x = _metric_tensor(candidate_frames)

    sharpness, contrast, exposure = _quality_metrics(x)
    sharp_n = _minmax(sharpness)
    contrast_n = _minmax(contrast)
    quality = 0.70 * sharp_n + 0.20 * contrast_n + 0.10 * exposure

    if x.shape[0] > 1:
        temporal_delta = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        temporal_delta[0] = (x[0] - x[1]).abs().mean()
        temporal_delta[-1] = (x[-1] - x[-2]).abs().mean()
        if x.shape[0] > 2:
            temporal_delta[1:-1] = 0.5 * (x[1:-1] - x[:-2]).abs().mean(dim=(1, 2, 3))
            temporal_delta[1:-1] += 0.5 * (x[1:-1] - x[2:]).abs().mean(dim=(1, 2, 3))
        stability = 1.0 - _minmax(temporal_delta)
    else:
        stability = torch.ones_like(quality)
    stable_quality = 0.80 * quality + 0.20 * stability

    similarity = None
    if source_image is not None:
        similarity = _similarity(x, source_image)

    effective_strategy = strategy
    strategy_warning = ""
    if strategy == "sharpest":
        scores = sharp_n
    elif strategy == "stable_quality":
        scores = stable_quality
    elif strategy == "most_similar_to_source":
        if similarity is None:
            scores = quality
            effective_strategy = "best_quality"
            strategy_warning = " WARNING: most_similar_to_source requires source_image; fell back to best_quality."
        else:
            scores = similarity
    elif strategy == "balanced_edit":
        if similarity is None:
            scores = stable_quality
            effective_strategy = "stable_quality"
            strategy_warning = " WARNING: balanced_edit requires source_image; fell back to stable_quality."
        else:
            sw = max(0.0, min(1.0, float(similarity_weight)))
            scores = sw * similarity + (1.0 - sw) * stable_quality
    else:
        scores = quality

    best_local = int(torch.argmax(scores).item())
    selected_index = int(candidate_indices[best_local].item())
    selected_score = float(scores[best_local].item())
    selected = frames[selected_index:selected_index + 1].clone()

    if emit_candidate_batch:
        k = min(max(1, int(top_k)), len(scores))
        top_local = torch.topk(scores, k=k, largest=True, sorted=True).indices
        top_global = candidate_indices[top_local].long()
        candidate_output = frames.index_select(0, top_global).clone()
        primary_output = frames.clone()
    else:
        candidate_output = frames.new_empty((0, *frames.shape[1:]))
        primary_output = selected

    sim_text = "n/a" if similarity is None else f"{float(similarity[best_local]):.4f}"
    report = (
        f"Selected frame {selected_index}/{n - 1}; requested_strategy={strategy}; "
        f"effective_strategy={effective_strategy}; score={selected_score:.4f}, "
        f"sharpness={float(sharp_n[best_local]):.4f}, quality={float(quality[best_local]):.4f}, "
        f"stability={float(stability[best_local]):.4f}, similarity={sim_text}; candidates={start}..{end - 1}."
        f"{strategy_warning}"
    )
    if emit_candidate_batch:
        report += (
            f" selected_image emits the complete {n}-frame decoded batch; "
            f"candidate_batch_debug emits the best {int(candidate_output.shape[0])} candidate(s), limited by top_k."
        )
    else:
        report += " Candidate batch suppressed."
    return primary_output, candidate_output, selected_index, selected_score, report


def _blur(image: torch.Tensor, radius: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    radius = min(int(radius), max(1, height - 1), max(1, width - 1))
    positions = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    sigma = max(1.0, radius / 3.0)
    kernel = torch.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padding_mode = "reflect" if height > radius and width > radius else "replicate"
    blurred = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode=padding_mode), horizontal, groups=channels)
    return F.conv2d(F.pad(blurred, (0, 0, radius, radius), mode=padding_mode), vertical, groups=channels)


def detail_tone_lock(
    source_image: torch.Tensor,
    refined_image: torch.Tensor,
    tone_lock: float,
    refinement_strength: float,
    detail_radius: int,
) -> torch.Tensor:
    source = source_image[..., :3].movedim(-1, 1).clone()
    refined = refined_image[..., :3].movedim(-1, 1).clone()
    if refined.shape[0] == 1 and source.shape[0] > 1:
        refined = refined.expand(source.shape[0], -1, -1, -1).clone()
    elif source.shape[0] != refined.shape[0]:
        refined = refined[:1].expand(source.shape[0], -1, -1, -1).clone()
    if source.shape[-2:] != refined.shape[-2:]:
        refined = F.interpolate(refined, size=source.shape[-2:], mode="bicubic", align_corners=False)
    source_low = _blur(source, detail_radius)
    refined_low = _blur(refined, detail_radius)
    tone_locked = refined + float(tone_lock) * (source_low - refined_low)
    output = source + float(refinement_strength) * (tone_locked - source)
    return output.clamp(0.0, 1.0).movedim(1, -1).clone()
