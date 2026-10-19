"""Cut one H3 continuation segment out of long inputs (A16 Part IV slice 2, L8.45). Original code, GPL-3.0.

A long plate / mask / control video arrives as ComfyUI's lazy VIDEO object: only the segment's frames are decoded, with
core's own float decode (VideoFromFile.as_trimmed + get_components). Frames are addressed by INDEX; core trims by
absolute presentation time, so the stream's start time is added here (a file whose first frame is not at t = 0 would
otherwise shift every slice by that offset). A long soundtrack (AUDIO) is cut on global time.
"""

from __future__ import annotations

from fractions import Fraction

import torch

FPS = 24
CONFORMABLE = (Fraction(24), Fraction(24000, 1001))


def frame_rate(video) -> Fraction:
    try:
        return Fraction(video.get_frame_rate()).limit_denominator(100000)
    except Exception:  # noqa: BLE001 - in-memory videos without a rate
        return Fraction(FPS)


def check_rate(video, what: str) -> tuple[Fraction, str]:
    """(rate, note). H3 renders 24 fps; a 23.976 plate is conformed frame for frame (reported); others are refused."""
    r = frame_rate(video)
    if r == Fraction(FPS):
        return r, ""
    if r in CONFORMABLE:
        return r, (f"{what} is {float(r):.3f} fps: taken frame for frame and played at 24 fps (0.1% faster); "
                   "its own sound cannot be kept (it would drift)")
    raise ValueError(f"H3 Continuation: {what} is {float(r):.3f} fps. H3 makes 24 fps video; convert it to 24 fps "
                     "(or 23.976, taken frame for frame) first.")


def stream_origin_seconds(video) -> float:
    """Presentation time of the first frame of the file behind a VIDEO (0 for in-memory videos)."""
    get = getattr(video, "get_stream_source", None)
    if get is None or type(video).__name__ != "VideoFromFile":
        return 0.0
    import av
    src = get()
    with av.open(src, mode="r") as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            return 0.0
        if vs.start_time is not None and vs.time_base is not None:
            return float(vs.start_time * vs.time_base)
        for f in c.decode(vs):
            return float(f.pts * vs.time_base) if f.pts is not None else 0.0
    return 0.0


def read_frames(video, start: int, count: int, *, with_audio: bool = False):
    """Frames [start, start + count) of a VIDEO as IMAGE [count, H, W, 3] (+ the window's sound).

    Returns (images, got, audio): `got` = frames that exist (< count at the end of the video; the rest are the last
    frame repeated, so the batch always has `count` frames). `audio` = the window's sound from frame `start`'s time,
    one frame longer than the window (margin for tick alignment), or None.
    """
    if count <= 0:
        raise ValueError("read_frames: count must be positive")
    rate = frame_rate(video)
    images, audio = None, None
    if type(video).__name__ == "VideoFromFile":     # lazy file: decode only the window
        t0 = stream_origin_seconds(video)
        window = video.as_trimmed(start_time=t0 + start / float(rate), duration=(count + 1) / float(rate),
                                  strict_duration=False)
        if window is not None:
            comp = window.get_components()
            images = comp.images
            audio = comp.audio if with_audio else None
    if images is None:                      # in-memory video (VideoFromComponents, ...): slice the decoded batch
        comp = video.get_components()
        images = comp.images[start:start + count + 1]
        if with_audio and comp.audio is not None:
            sr = int(comp.audio["sample_rate"])
            a = int(round(start / float(rate) * sr))
            n = int(round((count + 1) / float(rate) * sr))
            audio = {"waveform": comp.audio["waveform"][..., a:a + n], "sample_rate": sr}
    images = images[:count, ..., :3].float()
    got = int(images.shape[0])
    if 0 < got < count:
        images = torch.cat([images, images[-1:].expand(count - got, -1, -1, -1)], dim=0)
    return images, got, audio


def mask_from_frames(images: torch.Tensor) -> torch.Tensor:
    """A mask video's frames -> MASK [frames, H, W] (red channel, like core's Image To Mask 'red')."""
    return images[..., 0].clamp(0.0, 1.0).float()


def cut_audio(audio: dict, start_s: float, seconds: float, sr_out: int | None = None) -> dict:
    """[start_s, start_s + seconds) of an AUDIO on its own clock; silence before 0 and after the end."""
    wav = audio["waveform"]
    if wav.ndim == 2:
        wav = wav[None]
    sr = int(audio["sample_rate"])
    a = int(round(start_s * sr))
    n = int(round(seconds * sr))
    out = wav.new_zeros(wav.shape[0], wav.shape[1], n)
    lo, hi = max(a, 0), min(a + n, int(wav.shape[-1]))
    if hi > lo:
        out[..., lo - a:hi - a] = wav[..., lo:hi]
    return {"waveform": out, "sample_rate": sr if sr_out is None else sr_out}


def shift_audio(audio: dict, shift_s: float, seconds: float) -> dict:
    """Sound that starts at the window's start, re-cut to start `shift_s` later (earlier if negative)."""
    return cut_audio(audio, shift_s, seconds)


def audio_seconds(audio: dict) -> float:
    return int(audio["waveform"].shape[-1]) / float(audio["sample_rate"])
