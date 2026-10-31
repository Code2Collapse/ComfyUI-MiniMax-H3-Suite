"""H3 Proxy Manager (owner A21 Revision 3 Part 9.7; ledger L9.74 slice 7).

Writes a smaller copy of a lazy C2C video - 1/2, 1/4, 1/8 or a custom size - streamed through the vendored
StreamWriter (one chunk in memory), optionally with the selection's audio, and hands it back as a lazy handle on the
proxy file. The source map (frame ledger) says which source frame each proxy frame shows and how to scale
coordinates back. A proxy is cached: the same source (its fingerprint covers the file's stat, the selection, crop,
size and EXR layer) with the same settings is reused, never re-encoded; a changed source gets a new proxy.

Engine: the vendored C2C core (handle, reader, writer, timeline). Original node, GPL-3.0.
Design: docs/evidence/A21/phase1/DESIGN_slices7_10.md in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_vendor.c2c_core.timeline import ledger as L  # noqa: E402
from mmx_vendor.c2c_core.video import media as _media  # noqa: E402
from mmx_vendor.c2c_core.video import probe as _probe  # noqa: E402
from mmx_vendor.c2c_core.video import reader as _reader  # noqa: E402
from mmx_vendor.c2c_core.video.audio import lazy_audio_for_handle  # noqa: E402
from mmx_vendor.c2c_core.video.writer import CODECS, StreamWriter  # noqa: E402

from .media_window import C2C_VIDEO, is_handle  # noqa: E402
from .segment_io import job_folder  # noqa: E402

WHO = "H3 Proxy Manager"
SCALES = {"1/2": 2, "1/4": 4, "1/8": 8, "custom": 0}
EXT = {"h264": ".mp4", "prores": ".mov", "ffv1": ".mkv"}
MANIFEST = "h3.proxy_manifest"
_CHUNK = 8


def proxy_size(w: int, h: int, scale: str, cw: int, ch: int, codec: str) -> tuple[int, int, str]:
    """(width, height, note). Codecs that need an even size get one (rounded down, said in the note)."""
    div = SCALES.get(scale, 2)
    if div:
        pw, ph = max(1, round(w / div)), max(1, round(h / div))
    else:
        if cw <= 0 and ch <= 0:
            raise ValueError(f"{WHO}: scale 'custom' needs custom_width and / or custom_height")
        pw = cw if cw > 0 else max(1, round(w * ch / h))
        ph = ch if ch > 0 else max(1, round(h * cw / w))
    note = ""
    if CODECS[codec][4] and (pw % 2 or ph % 2):
        pw, ph = max(2, pw - pw % 2), max(2, ph - ph % 2)
        note = f"{codec} needs an even size: rounded down to {pw}x{ph}"
    return pw, ph, note


def _sha256(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def _timecode(media) -> str | None:
    try:
        from mmx_vendor.c2c_core.video import source_meta as SM
        return SM.record(media, media).get("timecode")
    except Exception:  # noqa: BLE001 - a timecode is a bonus
        return None


def encode_proxy(media, out_path: str, width: int, height: int, codec: str, audio: bool) -> dict:
    """Stream the selection at width x height into out_path (via a .partial file). Returns the writer's record."""
    small = media.scaled(width, height)
    aud = None
    if audio:
        aud = lazy_audio_for_handle(media)
        wf = aud["waveform"]                               # [1, C, N]
        rate, ch = int(aud["sample_rate"]), int(wf.shape[1])
    part = out_path + ".partial" + os.path.splitext(out_path)[1]
    try:
        w = StreamWriter(part, Fraction(media.fps).limit_denominator(1001000), width, height, codec=codec,
                         audio_rate=rate if audio else 0, audio_channels=ch if audio else 0)
        with w:
            for chunk in _reader.iter_chunks(small, _CHUNK, fmt="float32"):
                w.write(chunk[..., :3])
            if audio:
                w.write_audio(wf[0].numpy())
        rec = w.close()
        if os.path.exists(out_path):
            _reader.drain_container_pool()                # Windows: a pooled reader still holds the old proxy open
        os.replace(part, out_path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return rec


class MiniMaxH3_ProxyManager(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ProxyManager",
            display_name="H3 Proxy Manager",
            category="MiniMax H3/Media",
            description=(
                "Writes a smaller proxy of a lazy video (1/2, 1/4, 1/8 or custom; h264 / ProRes / FFV1; audio kept "
                "or not) into a job folder and returns it as a lazy handle. Cached: the same source with the same "
                "settings is reused, a changed source makes a new proxy. The source map takes proxy frames and "
                "coordinates back to the source."
            ),
            inputs=[
                C2C_VIDEO.Input("media", tooltip="From H3 Media Probe / Media Conform, or Load Video (C2C)."),
                io.Combo.Input("scale", options=list(SCALES), default="1/2"),
                io.Int.Input("custom_width", default=0, min=0, max=16384, tooltip="scale custom: 0 = keep aspect."),
                io.Int.Input("custom_height", default=0, min=0, max=16384, tooltip="scale custom: 0 = keep aspect."),
                io.Combo.Input("codec", options=list(EXT), default="h264",
                               tooltip="h264 (.mp4) small; prores (.mov) edit-friendly 10-bit; ffv1 (.mkv) lossless."),
                io.Combo.Input("audio", options=["keep", "none"], default="keep",
                               tooltip="keep: the selection's audio (AAC, FLAC in .mkv); a silent source gives silence."),
                io.String.Input("job_dir", default="h3_jobs/proxies",
                                tooltip="A folder inside ComfyUI's output folder; proxies go to <job_dir>/proxies/."),
                io.Boolean.Input("rebuild", default=False, tooltip="Encode again even when a matching proxy exists."),
            ],
            outputs=[
                C2C_VIDEO.Output("proxy", tooltip="Lazy handle on the proxy file."),
                io.String.Output("source_map", tooltip="h3.frame_ledger v1.1: proxy frame -> source frame (and "
                                                       "frame number); scale_x / scale_y back to the source."),
                io.String.Output("manifest", tooltip="h3.proxy_manifest v1: source, settings, file, sha256, status."),
                io.Image.Output("compare", tooltip="Source and proxy first frames, side by side at proxy size."),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, media, scale="1/2", custom_width=0, custom_height=0, codec="h264", audio="keep",
                job_dir="h3_jobs/proxies", rebuild=False) -> io.NodeOutput:
        import torch

        if not is_handle(media):
            raise ValueError(f"{WHO}: 'media' is not a C2C video handle - connect H3 Media Probe or Load Video (C2C).")
        if not len(media):
            raise ValueError(f"{WHO}: the media has no frames selected.")
        if codec not in EXT:
            raise ValueError(f"{WHO}: codec {codec!r} - use one of {', '.join(EXT)}")
        pw, ph, size_note = proxy_size(int(media.width), int(media.height), scale, int(custom_width),
                                       int(custom_height), codec)
        keep_audio = audio == "keep"
        settings = {"w": pw, "h": ph, "codec": codec, "audio": keep_audio}
        key = hashlib.sha256((media.fingerprint() + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:20]
        folder = job_folder(job_dir, create=True, who=WHO) / "proxies"
        folder.mkdir(parents=True, exist_ok=True)
        out_path = str(folder / f"proxy_{key}{EXT[codec]}")
        man_path = out_path + ".json"
        status = "cached"
        manifest = None
        if not rebuild and os.path.isfile(out_path) and os.path.isfile(man_path):
            try:
                manifest = json.loads(Path(man_path).read_text(encoding="utf-8"))
                if manifest.get("key") != key or manifest.get("sha256") != _sha256(out_path):
                    manifest = None
            except (OSError, ValueError):
                manifest = None
        if manifest is None:
            status = "rebuilt" if rebuild else "written"
            t0 = time.perf_counter()
            rec = encode_proxy(media, out_path, pw, ph, codec, keep_audio)
            src = media.source
            manifest = {
                "c2c_contract": MANIFEST, "c2c_contract_version": "1.0", "key": key,
                "source": {"fingerprint": media.fingerprint(), "kind": src.kind,
                           "path": getattr(src, "path", "") or getattr(src, "pattern", ""),
                           "width": int(media.width), "height": int(media.height), "frames": len(media),
                           "fps": str(Fraction(media.fps).limit_denominator(1001000)),
                           "timecode": _timecode(media)},
                "proxy": {"path": out_path, "width": pw, "height": ph, "codec": codec, "frames": rec["frames"],
                          "pix_fmt": rec["pix_fmt"], "audio": rec.get("audio")},
                "scale": {"x": pw / int(media.width), "y": ph / int(media.height)},
                "sha256": _sha256(out_path), "seconds": round(time.perf_counter() - t0, 3),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            Path(man_path).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        proxy = _probe.probe_file(out_path)
        sel = list(media.selected_indices())
        led = L.new(sel, fps=Fraction(media.fps).limit_denominator(1001000),
                    path=manifest["source"]["path"], total_source=int(media.frame_count),
                    numbers=_media.source_numbers(media))
        led["ops"] = [f"proxy {pw}x{ph} {codec}"]
        led["scale_x"] = pw / int(media.width)                      # proxy px * (1/scale) = source px
        led["scale_y"] = ph / int(media.height)
        a = _reader.read(media.scaled(pw, ph), [0], fmt="float32")[..., :3]
        b = _reader.read(proxy, [0], fmt="float32")[..., :3]
        compare = torch.cat([torch.from_numpy(a.copy()), torch.from_numpy(b.copy())], dim=2)
        report = "\n".join([
            f"status: {status} ({os.path.basename(out_path)})",
            f"source: {len(media)} frames, {media.width}x{media.height} @ {Fraction(media.fps)} fps",
            f"proxy:  {manifest['proxy']['frames']} frames, {pw}x{ph}, {codec}"
            + (f", audio {manifest['proxy']['audio']['codec']}" if manifest["proxy"].get("audio") else ", no audio"),
            "notes: " + ("; ".join(n for n in [size_note] if n) or "none"),
        ])
        return io.NodeOutput(proxy, json.dumps(led), json.dumps(dict(manifest, status=status)), compare, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3_ProxyManager": MiniMaxH3_ProxyManager}
