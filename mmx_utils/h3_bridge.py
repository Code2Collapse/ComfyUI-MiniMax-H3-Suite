# PORTED FROM: https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef@361624f h3_masked_bridge.py,
# h3_audio_grid.py, h3_workflow_utilities.py (MiniMaxH3AssembleBridgeAudio) and existing_video_extension.py (audio
# helpers), GPL-3.0. Logic kept; frames and sound are assembled in one step here, and the sources are resized to the
# decoded canvas when they differ.
"""Two-ended H3 AV bridge: generate the MIDDLE between two existing clips.

The last `preserve` frames of clip A and the first `preserve` frames of clip B are encoded into the target latent and
protected (noise mask 0); only the middle is generated. `preserve` must be a shared video / audio boundary,
39 + 51k frames (39 frames = 65 ticks, 51 frames = 85 ticks), so both streams cut at the same instant; the target is
5 + 17k frames and longer than both contexts together.

Core's VAE wrapper centre-crops audio input to a multiple of its hop (800 samples at 32 kHz) for the H3 audio VAE, so
every audio window here is cut to an exact multiple first - the seam can then never move.
"""

from __future__ import annotations

from fractions import Fraction

import torch

FPS = 24
AUDIO_HZ = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


class BridgeError(ValueError):
    pass


def pixel_frames(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def legal_video_run(n: int) -> bool:
    return n >= 5 and (n - 5) % 17 == 0


def check_timing(target: int, preserve: int) -> None:
    if not legal_video_run(target):
        raise BridgeError(f"the target is {target} frames; an H3 clip is 5 + 17k frames (5, 22, 39 ... 192 ...).")
    if preserve < 39 or (preserve - 39) % 51:
        raise BridgeError(f"preserve_frames is {preserve}; it must be a shared video / audio boundary: 39, 90, 141, "
                          "192 ... (39 + 51k).")
    if 2 * preserve >= target:
        raise BridgeError(f"the target ({target} frames) must be longer than both contexts together ({2 * preserve}).")


def sample_at(frame: int, sr: int) -> int:
    return int(round(int(frame) / FPS * int(sr)))


def stereo(waveform: torch.Tensor, label: str) -> torch.Tensor:
    if getattr(waveform, "ndim", 0) != 3:
        raise BridgeError(f"{label}: the waveform must be [batch, channels, samples], got "
                          f"{tuple(getattr(waveform, 'shape', ()))}")
    w = waveform[:1]
    if w.shape[1] == 1:
        return w.repeat(1, 2, 1)
    if w.shape[1] == 2:
        return w
    raise BridgeError(f"{label} has {w.shape[1]} channels; mix it down to stereo first (taking two channels could "
                      "lose part of the mix).")


def resample(waveform: torch.Tensor, src: int, dst: int, label: str) -> torch.Tensor:
    if int(src) == int(dst):
        return waveform
    try:
        import torchaudio  # noqa: WPS433
    except ImportError as exc:
        raise BridgeError(f"{label} is {src} Hz, {dst} Hz is needed and torchaudio is not installed") from exc
    return torchaudio.functional.resample(waveform, int(src), int(dst))


def conform(waveform: torch.Tensor, want: int, label: str, max_change: float = 0.005) -> torch.Tensor:
    """Fix a tiny duration mismatch (24 fps picture vs 40 Hz sound) by resampling by the tiny ratio, never by padding
    silence (that would put a gap at every seam); a large mismatch is refused, not repaired."""
    want, have = int(want), int(waveform.shape[-1])
    if have == want:
        return waveform
    if want <= 0 or have <= 0 or abs(want - have) / float(want) > max_change:
        raise BridgeError(f"{label} is {have} samples where the picture needs {want}: off by more than "
                          f"{max_change * 100:.1f}%, which is not 24 fps / 40 Hz rounding.")
    r = Fraction(want, have).limit_denominator(10000)
    try:
        import torchaudio  # noqa: WPS433
        out = torchaudio.functional.resample(waveform, int(r.denominator), int(r.numerator))
    except ImportError:
        out = torch.nn.functional.interpolate(waveform, size=want, mode="linear", align_corners=False)
    if int(out.shape[-1]) != want:
        out = torch.nn.functional.interpolate(out, size=want, mode="linear", align_corners=False)
    return out


def fit(waveform: torch.Tensor, want: int) -> torch.Tensor:
    have = int(waveform.shape[-1])
    if have >= want:
        return waveform[..., :want]
    return torch.nn.functional.pad(waveform, (0, want - have))


def canonical_audio(audio: dict | None, sr: int, frames: int, label: str) -> torch.Tensor:
    """Stereo at `sr`, exactly as long as `frames` of picture (silence if there is no audio)."""
    want = sample_at(frames, sr)
    if audio is None:
        return torch.zeros((1, 2, want))
    w = resample(stereo(audio["waveform"], label), int(audio["sample_rate"]), sr, label)
    return conform(w.float().cpu(), want, label)


def audio_hop(audio_vae) -> tuple[int, int]:
    sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    first = getattr(audio_vae, "first_stage_model", None)
    hop = getattr(first, "samples_per_latent", None) or getattr(first, "hop_length", None)
    if hop is None:
        if sr % AUDIO_HZ:
            raise BridgeError(f"the audio VAE's {sr} Hz is not a whole number of samples per 1/40 s tick")
        hop = sr // AUDIO_HZ
    if abs(sr / float(hop) - AUDIO_HZ) > 1e-6:
        raise BridgeError(f"the audio VAE runs at {sr / float(hop):.3f} ticks per second; H3 needs 40")
    return sr, int(hop)


def encode_audio_window(audio_vae, wave: torch.Tensor, frames: int, side: str, label: str) -> tuple[torch.Tensor, int]:
    """Encode the last (tail) or first (head) `frames` of sound as exactly round(frames * 40 / 24) ticks."""
    sr, hop = audio_hop(audio_vae)
    ticks = int(round(frames / FPS * AUDIO_HZ))
    need = ticks * hop
    if int(wave.shape[-1]) < need:
        raise BridgeError(f"{label}: the sound is shorter than the {ticks}-tick window")
    win = wave[..., -need:] if side == "tail" else wave[..., :need]
    z = audio_vae.encode(win.movedim(1, -1))
    if getattr(z, "ndim", 0) != 4 or int(z.shape[-1]) != ticks:
        raise BridgeError(f"{label}: {need} samples encoded to {tuple(getattr(z, 'shape', ()))}, expected {ticks} ticks "
                          "[B,C,2,T]. Connect the H3 audio VAE.")
    return z[:1], ticks


def resize_frames(images: torch.Tensor, width: int, height: int, crop: str, chunk: int = 32) -> torch.Tensor:
    import comfy.utils  # noqa: WPS433
    out = []
    for s in range(0, int(images.shape[0]), chunk):
        x = images[s:s + chunk, ..., :3].movedim(-1, 1)
        out.append(comfy.utils.common_upscale(x, width, height, "lanczos", crop).movedim(1, -1))
    return torch.cat(out, dim=0)


def encode_video_window(vae, frames: torch.Tensor, width: int, height: int, crop: str, n: int, label: str):
    z = vae.encode(resize_frames(frames, width, height, crop))
    if getattr(z, "ndim", 0) != 5 or pixel_frames(int(z.shape[2])) != n:
        raise BridgeError(f"{label}: {n} frames encoded to {tuple(getattr(z, 'shape', ()))}; refusing a seam that is "
                          "out of phase. Connect the H3 video VAE.")
    return z[:1], int(z.shape[2])


def middle_slice(target: int, preserve: int) -> tuple[int, int]:
    return preserve, target - preserve


def assemble(images: torch.Tensor, audio: dict, start_frames: torch.Tensor, start_audio: dict | None,
             end_frames: torch.Tensor, end_audio: dict | None, target: int, preserve: int, crop: str = "center"):
    """Clip A (all of it) + the generated middle + clip B (all of it), frames and sound, on absolute boundaries."""
    check_timing(target, preserve)
    if int(images.shape[0]) != target:
        raise BridgeError(f"the decoded bridge has {int(images.shape[0])} frames; the target is {target}. Decode the "
                          "whole sampled latent.")
    a_n, b_n = int(start_frames.shape[0]), int(end_frames.shape[0])
    if min(a_n, b_n) < preserve:
        raise BridgeError(f"each clip needs at least {preserve} frames; got {a_n} and {b_n}")
    h, w = int(images.shape[1]), int(images.shape[2])
    notes = []
    srcs = []
    for name, f in (("clip A", start_frames), ("clip B", end_frames)):
        if (int(f.shape[1]), int(f.shape[2])) != (h, w):
            notes.append(f"{name} resized {int(f.shape[2])}x{int(f.shape[1])} -> {w}x{h} ({crop})")
            f = resize_frames(f, w, h, crop)
        srcs.append(f[..., :3].to(images.device, images.dtype))
    lo, hi = middle_slice(target, preserve)
    frames = torch.cat([srcs[0], images[lo:hi, ..., :3], srcs[1]], dim=0)
    sr = int(audio["sample_rate"])
    wave = stereo(audio["waveform"], "bridge sound").float().cpu()
    target_ticks, ctx_ticks = round(target / FPS * AUDIO_HZ), preserve * 5 // 3
    tick = lambda t: int(round(t / AUDIO_HZ * sr))  # noqa: E731
    if abs(int(wave.shape[-1]) - tick(target_ticks)) > 2:
        raise BridgeError(f"the decoded bridge sound is {int(wave.shape[-1])} samples; its latent is {target_ticks} "
                          f"ticks = {tick(target_ticks)} samples. Decode the bridge's own audio.")
    middle = wave[..., tick(ctx_ticks):tick(target_ticks - ctx_ticks)]
    a_end = sample_at(a_n, sr)
    m_end = sample_at(a_n + target - 2 * preserve, sr)
    end = sample_at(a_n + target - 2 * preserve + b_n, sr)
    middle = conform(middle, m_end - a_end, "bridge middle sound", max_change=0.05)
    a_wave = fit(canonical_audio(start_audio, sr, a_n, "clip A sound"), a_end)
    b_wave = fit(canonical_audio(end_audio, sr, b_n, "clip B sound"), end - m_end)
    out = torch.cat([a_wave, middle, b_wave], dim=-1)
    if int(out.shape[-1]) != end:
        raise BridgeError("the sound's sample accounting failed")
    notes.insert(0, f"{a_n} + {hi - lo} generated + {b_n} = {int(frames.shape[0])} frames; sound {end} samples at "
                    f"{sr} Hz ({end / sr:.3f} s, exactly the picture)")
    return frames, {"waveform": out, "sample_rate": sr}, "\n".join(notes)
