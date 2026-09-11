# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) NikoDemon80 — ComfyUI-H3-Motion-Context
# PORTED FROM: ComfyUI-H3-Motion-Context :: nodes.py, probe_node.py @ third_party

"""Shared helpers for H3 motion-context chaining nodes."""

from __future__ import annotations

import gc
import logging
import os
import re

import comfy.utils
import folder_paths
import node_helpers
import numpy as np
import torch

from mmx_utils.motion_layout_contract import ensure as _ensure_layout_contract

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:
    _st_load = _st_save = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("mmx_motion_context")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)

ENCODE_MODE = "video"
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"
CROP = "disabled"

# seam probe thresholds (from seam_probe.py / level_step.py)
CORR_CREDIBLE = 0.6
FLOOR_CLEAN, FLOOR_MARGINAL = 0.25, 0.50
BROADBAND_OK = 0.40


def ensure_layout_ok(context: str = "") -> None:
    """Prove ComfyUI still places anchors the way this pack needs, once."""
    _ensure_layout_contract(context or "pinning a clip")


def pixel_frames(latent_t: int) -> int:
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def step_offsets(latent_t: int) -> list[int]:
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def resize(image, width: int, height: int, crop: str):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def encode_tail_audio(audio_vae, audio, seconds: float):
    """Encode the last `seconds` of a clip's audio with the H3 audio VAE."""
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "MiniMaxH3 Motion Context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning(
            "MiniMaxH3 Motion Context: context_audio is %.3fs, shorter than the "
            "%.3fs of pinned video. Pinning what there is.",
            have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams."""
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "MiniMaxH3 Motion Context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("MiniMaxH3 Motion Context: AV latent contains no streams")
    return parts


def video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "MiniMaxH3 Motion Context: expected video latent [B,C,T,H,W], "
            "got shape %s" % (tuple(video.shape),))
    return video


def steps_for_frames(n: int):
    """Latent steps covering exactly n pixel frames from cycle position 0."""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def video_tail_from_latent(latent, n: int):
    """Slice the last n pixel frames of video out of a generated H3 latent."""
    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "MiniMaxH3 Motion Context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 5, 22, "
            "39 or 56, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "MiniMaxH3 Motion Context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "MiniMaxH3 Motion Context: the %d step tail of a %d step latent starts "
            "at cycle position %d, not 0, so its frame spans would not match "
            "the positions written for them."
            % (steps, total, start % 5))
    covered = pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "MiniMaxH3 Motion Context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, step_offsets(steps), covered


def audio_tail_from_latent(latent, a_frames: int):
    """Slice the last `a_frames` worth of audio steps out of a generated H3 latent."""
    parts = streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "MiniMaxH3 Motion Context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            "MiniMaxH3 Motion Context: expected audio latent [B,C,2,T], "
            "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "MiniMaxH3 Motion Context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning(
            "MiniMaxH3 Motion Context: asked for %d audio steps, the latent "
            "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("MiniMaxH3 Motion Context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


def apply_motion_context(
    conditioning,
    vae,
    latent,
    context_length: int,
    audio_context_length: int = 24,
    context_frames=None,
    context_latent=None,
    audio_vae=None,
    context_audio=None,
):
    """Pin a run of consecutive frames from a previous clip into conditioning."""
    if context_latent is None and context_frames is None:
        return conditioning, 0, "no context wired; conditioning unchanged"

    encode_mode, anchor_mode = ENCODE_MODE, ANCHOR_MODE
    audio_mode, crop = AUDIO_MODE, CROP
    context_length = int(context_length)
    ensure_layout_ok()

    video = video_from_latent(latent)
    latent_t = int(video.shape[2])
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames(latent_t)

    if context_latent is not None:
        src_video = video_from_latent(context_latent)
        src_w = int(src_video.shape[4]) * 16
        src_h = int(src_video.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                "MiniMaxH3 Motion Context: context_latent is %dx%d but this "
                "clip is %dx%d. A latent cannot be resized."
                % (src_w, src_h, width, height))
        if int(src_video.shape[1]) != int(video.shape[1]):
            raise ValueError(
                "MiniMaxH3 Motion Context: context_latent has %d channels, "
                "this clip has %d."
                % (int(src_video.shape[1]), int(video.shape[1])))
        available = pixel_frames(int(src_video.shape[2]))
        video_src = "latent"
    else:
        available = int(context_frames.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("MiniMaxH3 Motion Context: no frames available to pin")
    if n < context_length:
        _LOG.warning(
            "MiniMaxH3 Motion Context: only %d frames available, pinning %d",
            available, n)

    if encode_mode == "video":
        run = next(g for g in VIDEO_RUN_GRID if g <= n)
        if run != n:
            _LOG.warning(
                "MiniMaxH3 Motion Context: %d frames is off the VAE grid; pinning "
                "the last %d instead (usable runs: 1, 5, 22, 39, 56)", n, run)
        n = run

    if n >= frame_count:
        raise ValueError(
            "MiniMaxH3 Motion Context: asked to pin %d frames into a %d frame clip."
            % (n, frame_count))

    if video_src == "latent" and steps_for_frames(n) is None:
        raise RuntimeError(
            "MiniMaxH3 Motion Context: a %d frame window is not a whole number "
            "of latent steps." % n)

    if video_src == "latent":
        blocks, offsets, covered = video_tail_from_latent(context_latent, n)
        span = covered
    else:
        tail = resize(context_frames[available - n:], width, height, crop)

    if video_src == "pixels" and encode_mode == "video":
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                "MiniMaxH3 Motion Context: video-mode encode returned shape %s."
                % (tuple(getattr(enc, "shape", ())),))
        steps = int(enc.shape[2])
        offsets = step_offsets(steps)
        covered = pixel_frames(steps)
        if covered != n:
            raise RuntimeError(
                "MiniMaxH3 Motion Context: %d frames encoded to %d latent steps "
                "covering %d frames; the VAE grid no longer matches VIDEO_RUN_GRID."
                % (n, steps, covered))
        blocks = [enc[:, :, k:k + 1] for k in range(steps)]
        span = covered
    elif video_src == "pixels":
        blocks, offsets = [], []
        for i in range(n):
            blocks.append(vae.encode(tail[i:i + 1]))
            offsets.append(i)
        span = n

    if anchor_mode == "before":
        indices = [o - span for o in offsets]
    else:
        indices = list(offsets)

    keyframes = []
    for p, blk in zip(indices, blocks):
        keyframes.append({
            "resolved_frame_index": p,
            "latent": blk,
        })

    ref_audio_t = 0
    audio_ref = None
    audio_kf = None
    audio_end_frame = None
    a_frames = 0
    audio_src = "off"
    if context_latent is not None or context_audio is not None:
        a_frames = int(audio_context_length) or span
        if context_latent is not None:
            if context_audio is not None:
                _LOG.info(
                    "MiniMaxH3 Motion Context: both context_latent and "
                    "context_audio wired; using the latent.")
            audio_latent, ref_audio_t, overhang = audio_tail_from_latent(
                context_latent, a_frames)
            audio_src = "latent"
        else:
            if audio_vae is None:
                raise ValueError(
                    "MiniMaxH3 Motion Context: context_audio supplied without "
                    "audio_vae.")
            audio_latent, ref_audio_t = encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS))
            overhang = 0.0
            audio_src = "vae"
        if audio_mode == "timeline":
            end_frame = float(span if anchor_mode == "head" else 0)
            end_frame += overhang / FRAME_RESCALE
            end_coord = round(FRAME_RESCALE * end_frame)
            end_frame = end_coord / FRAME_RESCALE
            audio_kf = {
                "resolved_frame_index": (end_frame
                                         - ref_audio_t / FRAME_RESCALE),
                "audio_latent": audio_latent,
            }
            audio_end_frame = end_frame
        else:
            audio_ref = {
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            }

    head_end = span if anchor_mode == "head" else 0
    tail_kfs = [audio_kf] if audio_kf is not None else []
    out = []
    dropped = []
    for emb, extra in conditioning:
        d = extra.copy()
        prior = d.get("minimax_keyframes") or []
        kept = []
        for kf in prior:
            p = kf.get("resolved_frame_index", 0)
            if p >= frame_count:
                raise ValueError(
                    "MiniMaxH3 Motion Context: keyframe at frame %s exceeds "
                    "clip length %d." % (p, frame_count))
            if p < head_end:
                dropped.append(p)
                continue
            kept.append(dict(kf))
        d["minimax_keyframes"] = kept + keyframes + tail_kfs
        out.append([emb, d])
    if dropped:
        _LOG.warning(
            "MiniMaxH3 Motion Context: dropped %d keyframe anchor(s) at "
            "frame(s) %s inside the pinned head.",
            len(dropped), sorted(set(dropped)))

    if audio_ref is not None:
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True)

    trim = span if anchor_mode == "head" else 0
    report = (
        "video from %s, %s/%s, %d frames -> %d cond blocks at indices %d..%d, "
        "%d frame clip at %dx%d, trim %d, audio %s"
        % (video_src, encode_mode, anchor_mode, n, len(blocks),
           indices[0], indices[-1], frame_count, width, height, trim,
           ("%d frames -> %d latent steps (%.3fs) from %s, %s"
            % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
               "on the timeline ending at frame %.3f" % audio_end_frame
               if audio_end_frame is not None
               else "stock ref placement"))
           if ref_audio_t else "off"))
    _LOG.info("MiniMaxH3 Motion Context: %s", report)
    return out, trim, report


def trim_clip(images, trim_frames: int, audio=None, fps: float = 24.0, match_tail: bool = True):
    """Remove leading pinned frames from decoded picture and sound."""
    n = max(0, int(trim_frames))
    total = int(images.shape[0])
    if n >= total:
        raise ValueError(
            "MiniMaxH3 Motion Context: asked to trim %d frames from a %d frame clip"
            % (n, total))
    out_images = images[n:] if n else images

    out_audio = audio
    report_lines = [f"trimmed {n} leading frames from {total}-frame clip"]
    if audio is not None:
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        seconds = n / float(fps)
        cut = int(round(seconds * sr))
        length = int(waveform.shape[-1])
        if cut >= length:
            raise ValueError(
                "MiniMaxH3 Motion Context: trimming %.3fs from %.3fs of audio would "
                "leave nothing. Check that fps matches the clip."
                % (seconds, length / sr))
        waveform = waveform[..., cut:]

        if match_tail:
            frames_left = total - n
            want = int(round(frames_left / float(fps) * sr))
            have = int(waveform.shape[-1])
            if have > want:
                over = have - want
                waveform = waveform[..., :want]
                report_lines.append(
                    "tail trimmed %d samples (%.2fms) to match %d frames"
                    % (over, over / sr * 1000.0, frames_left))
            elif have < want:
                missing = want - have
                waveform = torch.nn.functional.pad(waveform, (0, missing))
                report_lines.append(
                    "tail padded %d zero samples (%.2fms) to match %d frames"
                    % (missing, missing / sr * 1000.0, frames_left))

        out_audio = {"waveform": waveform, "sample_rate": sr}
        drift_ms = abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0
        report_lines.append(
            "%d frames / %.4fs picture, %.4fs sound, drift %.2fms"
            % (total - n, (total - n) / float(fps),
               int(waveform.shape[-1]) / sr, drift_ms))
    elif n:
        report_lines.append(
            "no audio wired; picture trimmed %.3fs ahead of any muxed sound"
            % (n / float(fps)))

    return out_images, out_audio, "\n".join(report_lines)


def resolve_latent_path(path, clip_index=0):
    """Turn the loader's path input into a concrete file."""
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_context"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            idx = int(clip_index)
            if idx <= 0:
                raise FileNotFoundError(
                    "MiniMaxH3 Motion Context: clip_index 0 does not load a file.")
            endings = ("_%05d.safetensors" % idx,
                       "_clip%03d.safetensors" % idx)
            files = [os.path.join(c, f) for f in os.listdir(c)
                     if f.endswith(endings)]
            if not files:
                near = [f for f in os.listdir(c)
                        if f.endswith("_%05d_.safetensors" % idx)]
                hint = ""
                if near:
                    hint = (
                        " Found %s (auto-numbered save; trailing underscore = "
                        "numbered by RUN)." % near[0])
                raise FileNotFoundError(
                    "MiniMaxH3 Motion Context: no saved latent for clip %d "
                    "(no *_%05d.safetensors in %s).%s"
                    % (idx, idx, c, hint))
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "MiniMaxH3 Motion Context: %r is neither a file nor a folder." % p)


def latent_folder(path):
    """Directory that holds this chain's slot files, or None."""
    p = (path or "").strip().strip('"').strip("'") or "h3_context"
    for c in (p, os.path.join(folder_paths.get_output_directory(), p)):
        if os.path.isdir(c):
            return c
        if os.path.isfile(c):
            return os.path.dirname(c)
    return None


def clip_slot_exists(latent_path, clip_index=1) -> bool:
    try:
        resolve_latent_path(latent_path, int(clip_index))
        return True
    except FileNotFoundError:
        return False


_CHAIN_SLOT_FILE = re.compile(
    r"(?:_\d{5}|_\d{5}_|_clip\d{3})\.safetensors(?:\.tmp)?$"
)


def clear_clip_slots(latent_path) -> int:
    """Delete numbered chain slots. Custom filenames are left alone."""
    folder = latent_folder(latent_path)
    if not folder:
        return 0
    removed = 0
    for name in os.listdir(folder):
        if not _CHAIN_SLOT_FILE.search(name):
            continue
        fp = os.path.join(folder, name)
        try:
            os.remove(fp)
        except OSError:
            gc.collect()
            os.remove(fp)
        removed += 1
    if removed:
        _LOG.info(
            "MiniMaxH3 Motion Context: cleared %d chain slot(s) from %s",
            removed, folder)
    return removed


def register_chain_routes() -> None:
    """No-op: JavaScript chain UI is not shipped in MiniMaxSuite."""
    return


def write_safetensors(path, tensors) -> None:
    tmp = path + ".tmp"
    try:
        _st_save(tensors, tmp, metadata={"format": "h3_motion_context_av_v1"})
        try:
            os.replace(tmp, path)
        except OSError:
            gc.collect()
            os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def save_context_latent(latent, filename_prefix: str, clip_index: int = 0):
    """Save an H3 AV latent for the next run's context_latent input."""
    if _st_save is None:
        raise RuntimeError(
            "MiniMaxH3 Motion Context: safetensors is not available.")
    parts = streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "MiniMaxH3 Motion Context: latent has no audio stream.")
    video = parts[0].cpu().contiguous()
    audio = parts[1].cpu().contiguous()
    folder, filename, counter, _, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory())
    if int(clip_index) > 0:
        path = os.path.join(folder, "%s_%05d.safetensors"
                            % (filename, int(clip_index)))
    else:
        path = os.path.join(folder, "%s_%05d_.safetensors"
                            % (filename, counter))
    write_safetensors(path, {"video": video, "audio": audio})
    report = "saved AV latent to %s (video %s, audio %s)" % (
        path, tuple(video.shape), tuple(audio.shape))
    _LOG.info("MiniMaxH3 Motion Context: %s", report)
    return path, report


def load_context_latent(latent_path: str, clip_index: int = 0):
    """Load a saved H3 AV latent for context_latent only."""
    if int(clip_index) <= 0:
        return None, "clip_index 0: no previous clip; output is empty"
    if _st_load is None:
        raise RuntimeError(
            "MiniMaxH3 Motion Context: safetensors is not available.")
    path = resolve_latent_path(latent_path, clip_index)
    data = _st_load(path)
    if "video" not in data or "audio" not in data:
        raise ValueError(
            "MiniMaxH3 Motion Context: %s is not an h3_motion_context latent." % path)
    video = data.pop("video").contiguous().clone()
    audio = data.pop("audio").contiguous().clone()
    report = "loaded AV latent from %s (video %s, audio %s)" % (
        path, tuple(video.shape), tuple(audio.shape))
    _LOG.info("MiniMaxH3 Motion Context: %s", report)
    return {"samples": [video, audio]}, report


# --- seam probe helpers (probe_node.py) ---


def mono(waveform):
    """Any audio tensor -> mono float64 numpy."""
    a = waveform
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 2:
        axis = 0 if a.shape[0] <= a.shape[1] else 1
        a = a.mean(axis=axis)
    return a


def norm_xcorr(win, ref):
    """Slide `win` across `ref`; return normalised correlation curve."""
    win = win - win.mean()
    ref = ref - ref.mean()
    wn = np.sqrt((win ** 2).sum())
    if wn < 1e-12:
        return None
    corr = np.correlate(ref, win, mode="valid")
    csum = np.concatenate(([0.0], np.cumsum(ref ** 2)))
    seg = csum[len(win):] - csum[:-len(win)]
    denom = wn * np.sqrt(np.maximum(seg, 1e-12))
    return corr / denom


def tracked_peak(ncc, expect_idx):
    """Pick alignment closest to previous window among local maxima."""
    best = float(ncc.max())
    i = np.arange(1, len(ncc) - 1)
    peaks = i[(ncc[i] >= ncc[i - 1]) & (ncc[i] >= ncc[i + 1]) &
              (ncc[i] >= 0.9 * best)]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(ncc))])
    pick = int(peaks[np.argmin(np.abs(peaks - expect_idx))])
    return pick, float(ncc[pick])


def rms(x):
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def floor_level(x, sr, win_ms=20.0, pct=10.0):
    """Low percentile of short-window RMS."""
    win = max(1, int(round(win_ms / 1000.0 * sr)))
    n = len(x) // win
    if n < 3:
        return rms(x)
    trimmed = x[:n * win].reshape(n, win)
    levels = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return float(np.percentile(levels, pct))


def step_ratio(a, b):
    denom = a + b
    if denom <= 1e-12:
        return 0.0
    return abs(a - b) / denom


def db(a, b):
    if a <= 1e-12 or b <= 1e-12:
        return float("inf")
    return abs(20.0 * np.log10(a / b))


def clip_a_audio(latent, audio_vae, sr, fps):
    """Decode clip A's audio and return (tail-matched, raw, frames)."""
    parts = streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "clip_a_latent has no audio stream. Wire the sampler "
            "output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    frames = pixel_frames(int(video.shape[2]))
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    decoded = audio_vae.decode(audio[:1])
    a = mono(decoded)
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if vae_sr != sr:
        raise ValueError(
            "clip A decodes at %d Hz but clip B is %d Hz. Not comparable."
            % (vae_sr, sr))
    want = int(round(frames / float(fps) * sr))
    have = len(a)
    raw = a
    if have > want:
        a = a[:want]
    elif have < want:
        a = np.concatenate([a, np.zeros(want - have)])
    return a, raw, frames


def continuation_report(a, b, sr, span, window_ms, search_ms):
    """B's first `span` samples should reconstruct A's last `span`."""
    win = max(1, int(round(window_ms / 1000.0 * sr)))
    hop = max(1, int(round(0.025 * sr)))
    search = max(1, int(round(search_ms / 1000.0 * sr)))
    a_tail_start = len(a) - span

    lags, corrs = [], []
    t, prev_lag = 0, None
    while t + win <= span:
        centre = a_tail_start + t
        lo = max(0, centre - search)
        hi = min(len(a), centre + win + search)
        ref = a[lo:hi]
        if len(ref) <= win:
            break
        ncc = norm_xcorr(b[t:t + win], ref)
        if ncc is None:
            t += hop
            continue
        if prev_lag is None:
            pick = int(np.argmax(ncc))
            corr = float(ncc[pick])
        else:
            expect = (centre - prev_lag) - lo
            pick, corr = tracked_peak(ncc, expect)
        lag = centre - (lo + pick)
        lags.append(lag / sr * 1000.0)
        corrs.append(corr)
        if corr > CORR_CREDIBLE:
            prev_lag = lag
        t += hop

    out = ["continuation over the pinned span"]
    if not corrs:
        out.append("  window longer than the span; nothing analysed")
        return out
    corrs = np.array(corrs)
    lags = np.array(lags)
    strong = corrs > CORR_CREDIBLE
    out.append("  mean corr   %.3f   (%d/%d windows above %.1f)"
               % (corrs.mean(), int(strong.sum()), len(corrs), CORR_CREDIBLE))
    if strong.any():
        sl = lags[strong]
        out.append("  lag         %+.2f ms mean, %+.2f to %+.2f "
                   "(credible windows only)"
                   % (sl.mean(), sl.min(), sl.max()))
        at_edge = np.abs(sl) >= (search / sr * 1000.0 - 1.0)
        if at_edge.any():
            out.append("  %d window(s) pegged at the search limit; the "
                       "real lag is outside it. Raise search_ms."
                       % int(at_edge.sum()))
    else:
        out.append("  lag         not reported: no window correlated "
                   "well enough to trust an alignment")
    if corrs.mean() < 0.3:
        out.append("  reading: the pinned audio is not being continued "
                   "at all. Check that context_latent or "
                   "context_audio is actually wired.")
    elif corrs.mean() < CORR_CREDIBLE:
        out.append("  reading: weak. The model is imitating the tail "
                   "rather than continuing it.")
    elif strong.any() and abs(lags[strong].mean()) > 12.0:
        out.append("  reading: locked but offset by more than half an "
                   "audio step. Placement, not continuation.")
    else:
        out.append("  reading: clean continuation.")
    return out


def level_report(a, b, sr, span):
    """The delivered join: A's tail against B's first post-trim audio."""
    b_delivered = b[span:]
    n = min(int(0.5 * sr), len(a), len(b_delivered))
    out = ["level step across the delivered cut"]
    if n < int(0.05 * sr):
        out.append("  too little audio either side of the cut to measure")
        return out
    left, right = a[-n:], b_delivered[:n]
    bb = step_ratio(rms(left), rms(right))
    fl = step_ratio(floor_level(left, sr), floor_level(right, sr))
    out.append("  broadband   %.3f  (%.1f dB)   %s"
               % (bb, db(rms(left), rms(right)),
                  "ok" if bb < BROADBAND_OK else "high"))
    out.append("  floor       %.3f  (%.1f dB)   %s"
               % (fl, db(floor_level(left, sr), floor_level(right, sr)),
                  "clean" if fl < FLOOR_CLEAN
                  else "marginal" if fl < FLOOR_MARGINAL
                  else "audible step"))
    out.append("  measured over %.0f ms either side" % (n / sr * 1000.0,))
    if fl >= FLOOR_MARGINAL and bb < BROADBAND_OK:
        out.append("  reading: the music matches but the bed underneath "
                   "restarts at the cut. Wire the previous clip's "
                   "audio into Motion Context.")
    out.append("  note: the floor figure reads the quiet moments "
               "between content, so wall to wall music with no rests "
               "gives it nothing to measure and it collapses onto the "
               "broadband number.")
    return out


def run_seam_probe(
    clip_b_untrimmed,
    trim_frames,
    clip_a_latent=None,
    audio_vae=None,
    fps=24.0,
    window_ms=50.0,
    search_ms=40.0,
):
    """Measure a chain join: continuation quality and level step."""
    passthrough = clip_b_untrimmed
    sr = int(clip_b_untrimmed["sample_rate"])
    b = mono(clip_b_untrimmed["waveform"])
    span_s = max(0, int(trim_frames)) / float(fps)
    span = int(round(span_s * sr))

    lines = ["H3 seam probe",
             "clip B (untrimmed): %.4fs at %d Hz" % (len(b) / sr, sr),
             "pinned span: %d frames = %.4fs = %d samples"
             % (int(trim_frames), span_s, span)]

    if span <= 0:
        lines += ["", "trim_frames is 0, so there is no pinned span "
                      "and nothing to measure. Wire it from the "
                      "Motion Context node."]
        return passthrough, "\n".join(lines)
    if len(b) < span:
        lines += ["", "clip B is shorter than the pinned span. This "
                      "is the TRIMMED audio; wire the VAE decode "
                      "output instead."]
        return passthrough, "\n".join(lines)
    if clip_a_latent is None or audio_vae is None:
        missing = "clip_a_latent" if clip_a_latent is None else "audio_vae"
        lines += ["", "no %s wired: measuring the join needs the "
                      "previous clip's latent and the audio VAE to "
                      "decode it." % missing]
        return passthrough, "\n".join(lines)

    try:
        a, raw, frames = clip_a_audio(clip_a_latent, audio_vae, sr, fps)
    except ValueError as exc:
        lines += ["", str(exc)]
        return passthrough, "\n".join(lines)

    pad = len(a) - len(raw)
    lines.append("clip A: %d frames, decoded %.4fs, tail-matched to "
                 "%.4fs (%s %d samples)"
                 % (frames, len(raw) / sr, len(a) / sr,
                    "padded" if pad > 0 else "trimmed" if pad < 0
                    else "exact", abs(pad)))
    lines.append("")
    if len(a) < span:
        lines.append("clip A is shorter than the pinned span, so "
                     "there is nothing to compare against.")
        return passthrough, "\n".join(lines)
    lines += continuation_report(a, b, sr, span, window_ms, search_ms)
    if pad > 0 and len(raw) >= span:
        alt = continuation_report(raw, b, sr, span, window_ms, search_ms)
        lines += ["", "cross-check: same audio anchored on the raw "
                      "decode instead of the padded end"]
        lines += ["  " + ln for ln in alt[1:]]
        lines.append("  the padding is %.2f ms, so these two should "
                     "differ by that much. Whichever reads near zero "
                     "is the true alignment."
                     % (pad / sr * 1000.0,))
    lines.append("")
    lines += level_report(a, b, sr, span)
    report = "\n".join(lines)
    for line in lines:
        _LOG.info("MiniMaxH3 Motion Context: %s", line)
    return passthrough, report
