# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/media.py (MIT, H3 Prompt Writer contributors)
#   (process_image sizing, process_video frame positions and seeking, _contact_sheet_columns, _build_contact_sheet)
# PORTED FROM: third_party/ComfyUI-MiniMaxH3-Prompt-Writer/backend/assembly.py (MIT, H3 Prompt Writer contributors)
#   (_media_line wording)
"""Node inputs -> numbered references: pictures, ordered video contact sheets, audio clips.

Changed here: the Writer prepared uploaded files in a session store; this node takes IMAGE tensors, a core VIDEO,
an IMAGE sequence and an AUDIO dict, and keeps everything in memory. A VIDEO backed by a file or stream is sampled
by seeking (the Writer's method) so a long plate is never decoded whole.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

from .modes import REFERENCE_LIMITS, WriterError

PREPARED_MAX_SIDE = 1536
FRAME_MAX_SIDE = 768
CONTACT_SHEET_INDEX_BASE_SIZE = 18
CONTACT_SHEET_INDEX_SCALE = 1.75
AUDIO_RATE = 16_000
AUDIO_MAX_SECONDS = 30.0
PLATE_AUDIO_SOURCE = "audio track of <Video 1>"


@dataclass
class Reference:
    kind: str                       # "image" | "video" | "audio"
    ref: str                        # "<Picture 1>" ...
    source: str                     # which node input it came from, e.g. "images[0]"
    width: int = 0
    height: int = 0
    duration: float | None = None
    timestamps: list[float] = field(default_factory=list)
    data: bytes = b""               # jpeg (picture / contact sheet) or wav (audio)
    mime: str = ""
    has_audio: bool = False
    sent: bool = False              # set by the node: the bytes went to the model

    def manifest_line(self) -> str:
        detail = self.kind
        if self.width and self.height:
            detail += f", {self.width}x{self.height}"
        if self.duration is not None:
            detail += f", {self.duration:g}s"
        if self.kind == "video":
            times = ", ".join(f"{t:g}s" for t in self.timestamps)
            if times:
                detail += f", sampled frames at {times}"
            if self.has_audio:
                detail += ", has its own audio track"
        elif self.kind == "audio" and self.source == PLATE_AUDIO_SOURCE:
            detail = "audio, the synchronized audio track of <Video 1>"
        elif self.kind == "audio":
            detail += (", attached as audio" if self.sent else
                       ", not analyzed by the model; role must come only from the user's brief")
        return f"{self.ref}: {self.source} ({detail})"


def _need(module: str):
    try:
        if module == "PIL":
            from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: F401
            import PIL
            return PIL
        if module == "numpy":
            import numpy
            return numpy
    except Exception as e:  # noqa: BLE001
        raise WriterError(f"{module} is needed to prepare the references and could not be imported ({e}).") from None
    raise ValueError(module)


def _to_pil(frame: Any):
    """[H,W,C] float 0-1 tensor or array -> RGB PIL image."""
    _need("PIL")
    np = _need("numpy")
    from PIL import Image
    arr = frame.detach().cpu().float().numpy() if hasattr(frame, "detach") else np.asarray(frame, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _jpeg(image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def image_to_jpeg(frame: Any) -> tuple[bytes, int, int]:
    """One picture, longest side <= 1536 (the Writer's prepared size). Returns (jpeg, width, height) of the input."""
    from PIL import Image
    image = _to_pil(frame)
    width, height = image.size
    prepared = image.copy()
    prepared.thumbnail((PREPARED_MAX_SIDE, PREPARED_MAX_SIDE), Image.Resampling.LANCZOS)
    return _jpeg(prepared, 92), width, height


def sample_times(duration: float, count: int) -> list[float]:
    """Ported frame positions (include_endpoints=True, sample_index 0)."""
    count = max(2, min(24, int(count)))
    margin = min(0.25, duration * 0.03)
    span = duration - 2 * margin
    positions = [0.0, *[min(0.999, max(0.001, index / (count - 1))) for index in range(1, count - 1)], 1.0]
    return [margin + span * position for position in positions]


def _contact_sheet_columns(frame_count: int) -> int:
    if frame_count <= 4:
        return 2
    return 3 if frame_count <= 6 else 4


def build_contact_sheet(frames: list) -> tuple[bytes, int, int]:
    """Ordered contact sheet from PIL frames -> (jpeg, width, height). Layout as the Writer's."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    columns = _contact_sheet_columns(len(frames))
    rows = math.ceil(len(frames) / columns)
    cell_width = 384
    source_width, source_height = frames[0].size
    cell_height = max(216, min(512, round(cell_width * source_height / max(source_width, 1))))
    gutter_height = 40
    cell_total_height = cell_height + gutter_height
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_total_height), "#101216")
    draw = ImageDraw.Draw(sheet)
    try:
        index_font = ImageFont.load_default(size=CONTACT_SHEET_INDEX_BASE_SIZE * CONTACT_SHEET_INDEX_SCALE)
    except TypeError:                     # Pillow < 10.1 has no size argument
        index_font = ImageFont.load_default()
    for index, image in enumerate(frames):
        fitted = ImageOps.contain(image.convert("RGB"), (cell_width, cell_height), Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        left = column * cell_width + (cell_width - fitted.width) // 2
        cell_top = row * cell_total_height
        top = cell_top + gutter_height + (cell_height - fitted.height) // 2
        sheet.paste(fitted, (left, top))
        draw.text((column * cell_width + 10, cell_top + 4), str(index + 1), font=index_font, fill=(183, 188, 198))
    return _jpeg(sheet, 90), sheet.size[0], sheet.size[1]


def _thumb(image):
    from PIL import Image
    image = image.convert("RGB")
    image.thumbnail((FRAME_MAX_SIDE, FRAME_MAX_SIDE), Image.Resampling.LANCZOS)
    return image


def frames_from_tensor(frames: Any, fps: float, count: int) -> tuple[list, list[float], float]:
    """IMAGE sequence [T,H,W,C] -> (sampled PIL frames, timestamps, duration)."""
    total = int(frames.shape[0])
    if total < 1:
        raise WriterError("video_frames is empty.")
    fps = float(fps) if fps and fps > 0 else 24.0
    duration = total / fps
    if total == 1:
        return [_thumb(_to_pil(frames[0]))], [0.0], duration
    picked, stamps = [], []
    for t in sample_times(duration, min(count, total)):
        idx = min(total - 1, max(0, int(round(t * fps))))
        picked.append(_thumb(_to_pil(frames[idx])))
        stamps.append(round(t, 3))
    return picked, stamps, duration


def _frames_by_seeking(source: Any, count: int) -> tuple[list, list[float], float, bool]:
    """Ported process_video seeking on a path or BytesIO. Raises on any failure (the caller falls back)."""
    import av
    if hasattr(source, "seek"):
        source.seek(0)
    frames, stamps = [], []
    with av.open(source) as container:
        video = next(iter(container.streams.video), None)
        if video is None or video.time_base is None:
            raise ValueError("no video stream")
        has_audio = any(True for _ in container.streams.audio)
        duration = None
        if video.duration is not None:
            duration = float(video.duration * video.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        if not duration or duration <= 0:
            raise ValueError("unknown duration")
        for timestamp in sample_times(duration, count):
            origin = float((video.start_time or 0) * video.time_base)
            target_time = origin + timestamp
            target_pts = max(0, int(target_time / float(video.time_base)))
            container.seek(target_pts, stream=video, backward=True, any_frame=False)
            selected = None
            for decoded in container.decode(video):
                selected = decoded
                frame_time = decoded.time
                if frame_time is None and decoded.pts is not None:
                    frame_time = float(decoded.pts * video.time_base)
                if frame_time is not None and frame_time >= target_time:
                    break
            if selected is None:
                continue
            frames.append(_thumb(selected.to_image()))
            stamps.append(round(timestamp, 3))
    if not frames:
        raise ValueError("no frames decoded")
    return frames, stamps, duration, has_audio


def frames_from_video(video: Any, count: int, notes: list[str]) -> tuple[list, list[float], float, bool, int, int]:
    """core VIDEO -> (frames, timestamps, duration, has_audio, width, height)."""
    width = height = 0
    try:
        width, height = video.get_dimensions()
    except Exception:  # noqa: BLE001
        pass
    try:
        source = video.get_stream_source()
        frames, stamps, duration, has_audio = _frames_by_seeking(source, count)
        return frames, stamps, duration, has_audio, width, height
    except Exception as e:  # noqa: BLE001 - components-only videos, odd containers
        notes.append(f"video: sampled from decoded components ({type(e).__name__}: {e})")
    comps = video.get_components()
    fps = float(getattr(comps, "frame_rate", 24) or 24)
    frames, stamps, duration = frames_from_tensor(comps.images, fps, count)
    has_audio = getattr(comps, "audio", None) is not None
    if not width:
        height, width = int(comps.images.shape[1]), int(comps.images.shape[2])
    return frames, stamps, duration, has_audio, width, height


def audio_to_wav(audio: dict, max_seconds: float = AUDIO_MAX_SECONDS) -> tuple[bytes, float]:
    """AUDIO {"waveform": [B,C,T], "sample_rate"} -> mono 16 kHz PCM16 WAV (first max_seconds), full duration."""
    import wave
    np = _need("numpy")
    wf = audio.get("waveform") if isinstance(audio, dict) else None
    sr = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
    if wf is None or sr <= 0:
        raise WriterError("The audio input is not a ComfyUI AUDIO (waveform + sample_rate).")
    arr = wf.detach().cpu().float().numpy() if hasattr(wf, "detach") else np.asarray(wf, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr
    duration = mono.shape[-1] / sr
    mono = mono[: int(max_seconds * sr)]
    if sr != AUDIO_RATE and mono.size > 1:
        n_out = max(1, int(round(mono.size * AUDIO_RATE / sr)))
        mono = np.interp(np.linspace(0, mono.size - 1, n_out), np.arange(mono.size), mono)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue(), round(duration, 3)


def collect_references(images=None, images_2=None, images_3=None, video=None, video_frames=None,
                       video_frames_fps: float = 24.0, audio=None, video_samples: int = 6,
                       notes: list[str] | None = None) -> list[Reference]:
    """Number every connected input: <Picture 1..> (images, images_2, images_3 batch items in order),
    <Video 1> (video), next <Video N> (video_frames), <Audio 1> (audio)."""
    notes = notes if notes is not None else []
    refs: list[Reference] = []
    n_pic = 0
    for name, batch in (("images", images), ("images_2", images_2), ("images_3", images_3)):
        if batch is None:
            continue
        if getattr(batch, "ndim", 4) == 3:
            batch = batch.unsqueeze(0)
        for i in range(int(batch.shape[0])):
            n_pic += 1
            if n_pic > REFERENCE_LIMITS["image"]:
                raise WriterError(f"Too many pictures: at most {REFERENCE_LIMITS['image']} (images, images_2 and "
                                  f"images_3 together).")
            data, w, h = image_to_jpeg(batch[i])
            refs.append(Reference("image", f"<Picture {n_pic}>", f"{name}[{i}]", w, h, data=data, mime="image/jpeg"))
    n_vid = 0
    if video is not None:
        n_vid += 1
        frames, stamps, duration, has_audio, w, h = frames_from_video(video, video_samples, notes)
        data, _sw, _sh = build_contact_sheet(frames)
        refs.append(Reference("video", f"<Video {n_vid}>", "video", w, h, round(duration, 3), stamps, data,
                              "image/jpeg", has_audio))
    if video_frames is not None:
        n_vid += 1
        frames, stamps, duration = frames_from_tensor(video_frames, video_frames_fps, video_samples)
        data, _sw, _sh = build_contact_sheet(frames)
        refs.append(Reference("video", f"<Video {n_vid}>", "video_frames", int(video_frames.shape[2]),
                              int(video_frames.shape[1]), round(duration, 3), stamps, data, "image/jpeg"))
    if audio is not None:
        data, duration = audio_to_wav(audio)
        refs.append(Reference("audio", "<Audio 1>", "audio", duration=duration, data=data, mime="audio/wav"))
    if len(refs) > REFERENCE_LIMITS["total"]:
        raise WriterError(f"Too many references: at most {REFERENCE_LIMITS['total']} in total.")
    return refs


def add_plate_audio(refs: list[Reference]) -> Reference | None:
    """V2V edit / Replacement / Continuation: <Video 1>'s own sound becomes the next <Audio N> (guide: "an enabled
    synchronized audio track from a reference video"). Manifest only: its bytes are not sent to the model."""
    plate = next((r for r in refs if r.ref == "<Video 1>" and r.has_audio), None)
    if plate is None:
        return None
    n = sum(1 for r in refs if r.kind == "audio") + 1
    if n > REFERENCE_LIMITS["audio"] or len(refs) >= REFERENCE_LIMITS["total"]:
        return None
    ref = Reference("audio", f"<Audio {n}>", PLATE_AUDIO_SOURCE, duration=plate.duration)
    refs.append(ref)
    return ref
