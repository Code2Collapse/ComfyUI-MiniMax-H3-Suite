"""A29 HDR file I/O (owner order A29, 2026-11-16; design docs/hdr/A29_DESIGN.md in the C2C workspace).

- EXR read / write: OpenCV (OPENCV_IO_ENABLE_OPENEXR is set before cv2 is imported) -> OpenImageIO -> OpenEXR,
  else a plain-English error. half or float32, ZIP compression. Linear light is never clamped.
- 16-bit PNG / TIFF of a log image (the log signal is clipped to 0..1 - allowed: it is a code, not light).
- HDR10: scene-linear Rec.709 -> Rec.2020 (hdr_math.gamut_convert) -> nits (1.0 = paper white, 203) -> PQ
  (hdr_math.pq_encode) -> Y'CbCr 4:2:0 10-bit (hdr_deliver.ycbcr10) -> ffmpeg pipe -> libx265 yuv420p10le with BT.2020 / SMPTE 2084 tags + HDR10 SEI.
  PQ holds at most 10000 nits; brighter pixels are clipped by the format and the share is reported.
- Read-back: ffprobe (else PyAV) for the tags, PyAV for a decoded frame's PQ signal.
No new dependency: cv2, OpenImageIO, PyAV and imageio-ffmpeg are each optional and guarded.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")          # must precede the first `import cv2`

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

try:
    from . import hdr_math as HM
except ImportError:                                              # imported as top-level mmx_utils.hdr_a29_io
    from mmx_utils import hdr_math as HM

HALF_MAX = 65504.0
_BACKENDS: list | None = None


# ------------------------------------------------------------------------------------------------------- backends
def _cv2():
    try:
        import cv2
        return cv2
    except Exception:  # noqa: BLE001
        return None


def _oiio():
    try:
        import OpenImageIO as oiio
        return oiio
    except Exception:  # noqa: BLE001
        return None


def _openexr():
    try:
        import OpenEXR  # noqa: F401
        import Imath  # noqa: F401
        return OpenEXR
    except Exception:  # noqa: BLE001
        return None


def _cv2_exr_works() -> bool:
    cv2 = _cv2()
    if cv2 is None:
        return False
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "probe.exr")
            px = np.full((2, 2, 3), 5.0, np.float32)
            if not cv2.imwrite(p, px, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
                return False
            back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            return back is not None and abs(float(back.max()) - 5.0) < 1e-6
    except Exception:  # noqa: BLE001
        return False


def exr_backends() -> list[str]:
    """Working EXR backends, best first: 'cv2' (only if a real EXR round trip succeeds), 'oiio', 'openexr'."""
    global _BACKENDS
    if _BACKENDS is None:
        out = []
        if _cv2_exr_works():
            out.append("cv2")
        if _oiio() is not None:
            out.append("oiio")
        if _openexr() is not None:
            out.append("openexr")
        _BACKENDS = out
    return list(_BACKENDS)


def _pick(backend: str) -> str:
    have = exr_backends()
    if backend != "auto":
        if backend not in have:
            raise RuntimeError(f"EXR backend '{backend}' is not available here (available: {have or 'none'})")
        return backend
    if not have:
        raise RuntimeError("No EXR writer available: OpenCV's EXR support is disabled and neither OpenImageIO nor "
                           "OpenEXR is installed. Install OpenImageIO (pip install OpenImageIO) or OpenEXR.")
    return have[0]


# ------------------------------------------------------------------------------------------------------------ EXR
def _as_rgb(rgb) -> np.ndarray:
    a = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] < 3:
        raise ValueError(f"expected an [H, W, 3] picture, got shape {tuple(a.shape)}")
    return np.ascontiguousarray(a[..., :3], dtype=np.float32)


def write_exr(path, rgb, precision: str = "half", backend: str = "auto") -> str:
    """Write one RGB EXR (ZIP compression). Values are written as they are (no clamp); returns the backend used."""
    if precision not in ("half", "float32"):
        raise ValueError("precision must be 'half' or 'float32'")
    px = _as_rgb(rgb)
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    b = _pick(backend)
    if b == "cv2":
        cv2 = _cv2()
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if precision == "half" else cv2.IMWRITE_EXR_TYPE_FLOAT,
                  cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
        if not cv2.imwrite(path, np.ascontiguousarray(px[..., ::-1]), params):
            raise RuntimeError(f"OpenCV could not write {path}")
    elif b == "oiio":
        oiio = _oiio()
        spec = oiio.ImageSpec(px.shape[1], px.shape[0], 3, oiio.HALF if precision == "half" else oiio.FLOAT)
        spec.attribute("compression", "zip")
        out = oiio.ImageOutput.create(path)
        if out is None:
            raise RuntimeError(f"OpenImageIO could not create {path}: {oiio.geterror()}")
        try:
            if not out.open(path, spec) or not out.write_image(px):
                raise RuntimeError(f"OpenImageIO could not write {path}: {out.geterror()}")
        finally:
            out.close()
    else:
        import Imath
        OpenEXR = _openexr()
        pt = Imath.PixelType(Imath.PixelType.HALF if precision == "half" else Imath.PixelType.FLOAT)
        hdr = OpenEXR.Header(px.shape[1], px.shape[0])
        hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        hdr["channels"] = {c: Imath.Channel(pt) for c in "RGB"}
        np_t = np.float16 if precision == "half" else np.float32
        f = OpenEXR.OutputFile(path, hdr)
        try:
            f.writePixels({c: px[..., i].astype(np_t).tobytes() for i, c in enumerate("RGB")})
        finally:
            f.close()
    return b


def read_exr(path, backend: str = "auto") -> np.ndarray:
    """One EXR -> float32 [H, W, 3] RGB (gray -> 3 channels, alpha dropped). No clamp."""
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path}")
    b = _pick(backend)
    if b == "cv2":
        cv2 = _cv2()
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise RuntimeError(f"OpenCV could not read {path}")
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, -1)
        else:
            a = a[..., :3][..., ::-1]
    elif b == "oiio":
        oiio = _oiio()
        inp = oiio.ImageInput.open(path)
        if inp is None:
            raise RuntimeError(f"OpenImageIO could not read {path}: {oiio.geterror()}")
        try:
            a = inp.read_image(oiio.FLOAT)
        finally:
            inp.close()
        if a is None:
            raise RuntimeError(f"OpenImageIO could not read the pixels of {path}")
        a = np.asarray(a)
        if a.ndim == 2:
            a = a[..., None]
        a = np.repeat(a, 3, -1) if a.shape[-1] == 1 else a[..., :3]
    else:
        import Imath
        f = _openexr().InputFile(path)
        try:
            dw = f.header()["dataWindow"]
            w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
            ft = Imath.PixelType(Imath.PixelType.FLOAT)
            a = np.stack([np.frombuffer(f.channel(c, ft), np.float32).reshape(h, w) for c in "RGB"], -1)
        finally:
            f.close()
    return np.ascontiguousarray(a, dtype=np.float32)


def exr_header(path) -> dict | None:
    """{'compression', 'pixel_type' ('half' | 'float')} from the file header via OpenImageIO; None without it."""
    oiio = _oiio()
    if oiio is None:
        return None
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        return None
    try:
        spec = inp.spec()
        comp = spec.getattribute("compression")
        fmt = str(spec.format)
    finally:
        inp.close()
    return {"compression": str(comp).lower() if comp is not None else None,
            "pixel_type": "half" if "half" in fmt.lower() else ("float" if "float" in fmt.lower() else fmt)}


# ------------------------------------------------------------------------------------------------ 16-bit PNG / TIFF
def to_uint16(code) -> np.ndarray:
    """Log code 0..1 -> uint16 (the log SIGNAL is clipped to 0..1; linear light never goes through here)."""
    return np.round(np.clip(_as_rgb(code), 0.0, 1.0) * 65535.0).astype(np.uint16)


def write_16bit(path, code) -> None:
    """16-bit PNG or TIFF (by extension) of a log image."""
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is needed to write 16-bit PNG / TIFF and is not installed")
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, np.ascontiguousarray(to_uint16(code)[..., ::-1])):
        raise RuntimeError(f"OpenCV could not write {path}")


def read_16bit(path) -> np.ndarray:
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is needed to read 16-bit PNG / TIFF and is not installed")
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"could not read {path}")
    if a.dtype != np.uint16:
        raise RuntimeError(f"{path} is not a 16-bit image ({a.dtype})")
    a = np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a[..., :3][..., ::-1]
    return a.astype(np.float32) / 65535.0


# --------------------------------------------------------------------------------------------------------- HDR10
MASTER_DISPLAY_2020 = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"    # BT.2020 primaries, D65, x50000
EXPECTED_TAGS = {"codec": "hevc", "pix_fmt": "yuv420p10le", "color_primaries": "bt2020",
                 "color_transfer": "smpte2084", "color_space": "bt2020nc"}


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        raise RuntimeError("ffmpeg not found: put ffmpeg on PATH or install imageio-ffmpeg") from None


def ffprobe_exe() -> str | None:
    return shutil.which("ffprobe")


def hdr10_signal(lin: torch.Tensor, paper_white: float = HM.PAPER_WHITE_NITS):
    """Scene-linear Rec.709 [T, H, W, 3] (1.0 = diffuse white) -> (PQ signal 0..1 in Rec.2020, info)."""
    rgb = HM.gamut_convert(lin[..., :3].float(), "rec709", "rec2020")
    negative = float((rgb < 0).any(-1).float().mean())
    nits = rgb.clamp_min(0.0) * float(paper_white)
    peak = nits.amax(-1)
    info = {
        "paper_white_nits": float(paper_white),
        "clipped_fraction": float((peak > 10000.0).float().mean()),
        "negative_fraction": negative,
        "max_nits": float(peak.max()) if peak.numel() else 0.0,
        "max_cll": int(math.ceil(min(float(peak.max()) if peak.numel() else 0.0, 10000.0))),
        "max_fall": int(math.ceil(min(float(peak.clamp(max=10000.0).mean(dim=(1, 2)).max()) if peak.numel() else 0.0,
                                      10000.0))),
    }
    return HM.pq_encode(nits), info


def _deliver():
    try:
        from . import hdr_deliver as D
    except ImportError:
        from mmx_utils import hdr_deliver as D
    return D


def yuv420p10(pq_rgb: np.ndarray) -> bytes:
    """PQ R'G'B' [H, W, 3] (even sides) -> raw yuv420p10le bytes: BT.2020 NCL narrow range via hdr_deliver.ycbcr10
    (exact code values; ffmpeg's swscale RGB->YUV was 1-2 codes off, measured), chroma = 2x2 mean."""
    y, cb, cr = _deliver().ycbcr10(np.clip(pq_rgb.astype(np.float64), 0.0, 1.0))
    h, w = y.shape

    def sub(c):
        return np.round(c.astype(np.float64).reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))).astype("<u2")

    return y.astype("<u2").tobytes() + sub(cb).tobytes() + sub(cr).tobytes()


def write_hdr10(path, lin: torch.Tensor, fps: float = 24.0, crf: int = 12, paper_white: float = HM.PAPER_WHITE_NITS,
                master_nits: int = 1000) -> dict:
    """Linear Rec.709 frames -> HDR10 HEVC .mp4 (PQ, Rec.2020, 10-bit 4:2:0, mastering display + MaxCLL SEI)."""
    if lin.ndim == 3:
        lin = lin[None]
    t, h, w = int(lin.shape[0]), int(lin.shape[1]), int(lin.shape[2])
    if t == 0:
        raise ValueError("no frames to write")
    pq, info = hdr10_signal(lin, paper_white)
    ph, pw = h % 2, w % 2
    if ph or pw:                                                  # 4:2:0 needs even sides: replicate the last row / col
        pq = torch.nn.functional.pad(pq.permute(0, 3, 1, 2), (0, pw, 0, ph), mode="replicate").permute(0, 2, 3, 1)
    H2, W2 = h + ph, w + pw
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    x265 = (f"crf={int(crf)}:hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            f"range=limited:master-display={MASTER_DISPLAY_2020}L({int(master_nits) * 10000},1):"
            f"max-cll={info['max_cll']},{info['max_fall']}:log-level=error")
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W2}x{H2}", "-r", f"{float(fps):g}", "-i", "-",
           "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
           "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv",
           "-x265-params", x265, path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err = b""
    try:
        for i in range(t):
            proc.stdin.write(yuv420p10(pq[i].cpu().numpy()))
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait(timeout=600)
    except (BrokenPipeError, OSError):
        err = proc.stderr.read() if proc.stderr else b""
        proc.kill()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError(f"ffmpeg failed to write the HDR10 file (exit {proc.returncode}): "
                           f"{err.decode('utf-8', 'replace')[-1500:]}")
    info.update(path=path, width=w, height=h, frames=t, padded=bool(ph or pw), encoded_width=W2, encoded_height=H2,
                fps=float(fps), crf=int(crf), master_display_nits=int(master_nits))
    return info


def probe_video(path) -> dict:
    """Tags of the first video stream: codec, pix_fmt, color_primaries / transfer / space / range, size, frames."""
    path = str(path)
    exe = ffprobe_exe()
    if exe:
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_streams", "-of", "json",
                            path], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
            return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"),
                    "color_primaries": s.get("color_primaries"), "color_transfer": s.get("color_transfer"),
                    "color_space": s.get("color_space"), "color_range": s.get("color_range"),
                    "width": s.get("width"), "height": s.get("height"),
                    "frames": int(s["nb_read_frames"]) if str(s.get("nb_read_frames", "")).isdigit() else None}
    try:
        import av
    except Exception:  # noqa: BLE001
        raise RuntimeError("cannot check the video: neither ffprobe nor PyAV is available") from None
    names_p = {1: "bt709", 9: "bt2020"}
    names_t = {1: "bt709", 16: "smpte2084", 18: "arib-std-b67"}
    names_s = {1: "bt709", 9: "bt2020nc", 10: "bt2020c"}
    with av.open(path) as c:
        st = c.streams.video[0]
        cc = st.codec_context
        n = sum(1 for _ in c.decode(st))
        return {"codec": cc.name, "pix_fmt": cc.pix_fmt, "color_primaries": names_p.get(cc.color_primaries),
                "color_transfer": names_t.get(cc.color_trc), "color_space": names_s.get(cc.colorspace),
                "color_range": {1: "tv", 2: "pc"}.get(cc.color_range), "width": cc.width, "height": cc.height,
                "frames": n}


def check_hdr10_tags(tags: dict) -> list[str]:
    """Problems (empty = an HDR10 file): every EXPECTED_TAGS entry must match."""
    return [f"{k} is {tags.get(k)!r}, expected {v!r}" for k, v in EXPECTED_TAGS.items() if tags.get(k) != v]


def decode_pq_frame(path, index: int = 0) -> np.ndarray:
    """Decoded PQ R'G'B' signal (0..1, float64 [H, W, 3]) of one frame of a 10-bit BT.2020 NCL narrow-range video.
    Chroma is upsampled by repetition (exact on flat areas)."""
    try:
        import av
    except Exception:  # noqa: BLE001
        raise RuntimeError("PyAV (av) is needed to decode the HDR10 file for the check") from None
    D = _deliver()
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i < index:
                continue
            if fr.format.name != "yuv420p10le":
                raise RuntimeError(f"expected yuv420p10le frames, got {fr.format.name}")
            w, h = fr.width, fr.height
            planes = []
            for k, pl in enumerate(fr.planes):
                pw, ph = (w, h) if k == 0 else ((w + 1) // 2, (h + 1) // 2)
                a = np.frombuffer(bytes(pl), np.uint16).reshape(ph, pl.line_size // 2)[:, :pw]
                if k:
                    a = np.repeat(np.repeat(a, 2, 0), 2, 1)[:h, :w]
                planes.append(a)
            return D.rgb_from_ycbcr10(np.stack(planes))
    raise IndexError(f"frame {index} not in {path}")


__all__ = ["exr_backends", "write_exr", "read_exr", "exr_header", "to_uint16", "write_16bit", "read_16bit",
           "ffmpeg_exe", "ffprobe_exe", "hdr10_signal", "write_hdr10", "probe_video", "check_hdr10_tags",
           "decode_pq_frame", "EXPECTED_TAGS", "HALF_MAX"]
