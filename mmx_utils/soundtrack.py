"""Replace a lazy video's soundtrack (ledger L9.79, Lip Sync Director): the video stream is COPIED packet for packet
into a Matroska file in the output folder (no re-encode, the same frames at the same times) and a new FLAC audio
stream is written that starts at the first kept frame's time and lasts exactly as long as the kept frames - cut or
padded with silence, said in the report. The handle then points at that file: every window, the lazy audio and the
job's final mux use the new sound, with no change anywhere else. Original code, GPL-3.0."""
from __future__ import annotations

import dataclasses
import os
from fractions import Fraction

import numpy as np


def _video_stream(container):
    vs = [s for s in container.streams if s.type == "video"]
    if not vs:
        raise ValueError(f"no video stream in {container.name}")
    return vs[0]


def remux_with_sound(src_path: str, video_stream_index: int, wave: np.ndarray, rate: int, start_sec: float,
                     duration_sec: float, out_path: str) -> dict:
    """Copy the video stream of ``src_path`` and add ``wave`` ([channels, samples] float, -1..1) as the only audio,
    beginning at ``start_sec`` (absolute, the source's own clock) for exactly ``duration_sec``."""
    import av

    want = max(0, int(round(duration_sec * rate)))
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim == 1:
        w = w[None]
    w = np.clip(w[:2], -1.0, 1.0)                                 # FLAC: mono or stereo
    have = int(w.shape[1])
    if have >= want:
        body, note = w[:, :want], (f"the new sound was cut by {(have - want) / rate:.3f} s" if have > want else
                                   "the new sound fits exactly")
    else:
        body = np.concatenate([w, np.zeros((w.shape[0], want - have), np.float32)], axis=1)
        note = f"the new sound was padded with {(want - have) / rate:.3f} s of silence"
    layout = "mono" if body.shape[0] == 1 else "stereo"
    tmp = out_path + ".part"
    src = av.open(src_path)
    try:
        vin = [s for s in src.streams if s.type == "video"][int(video_stream_index)] \
            if video_stream_index < len([s for s in src.streams if s.type == "video"]) else _video_stream(src)
        out = av.open(tmp, mode="w", format="matroska")
        try:
            try:
                vout = out.add_stream_from_template(vin)
            except AttributeError:                                   # older PyAV
                vout = out.add_stream(template=vin)
            aout = out.add_stream("flac", rate=int(rate), layout=layout)
            packets = 0
            for pkt in src.demux(vin):
                if pkt.dts is None and pkt.pts is None:
                    continue
                pkt.stream = vout
                out.mux(pkt)
                packets += 1
            rs = av.AudioResampler(format=aout.codec_context.format.name, layout=layout, rate=int(rate))
            first = int(round(start_sec * rate))
            pts, step = first, 4096
            for a in range(0, body.shape[1], step):
                fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(body[:, a:a + step]), format="fltp", layout=layout)
                fr.sample_rate = int(rate)
                for g in rs.resample(fr):
                    g.pts, g.time_base = pts, Fraction(1, int(rate))
                    pts += g.samples
                    for p in aout.encode(g):
                        out.mux(p)
            for g in rs.resample(None):
                g.pts, g.time_base = pts, Fraction(1, int(rate))
                pts += g.samples
                for p in aout.encode(g):
                    out.mux(p)
            for p in aout.encode(None):
                out.mux(p)
        finally:
            out.close()
    finally:
        src.close()
    os.replace(tmp, out_path)
    return {"path": out_path, "video_packets": packets, "audio_seconds": round(want / rate, 4),
            "start_sec": round(start_sec, 6), "rate": int(rate), "channels": int(body.shape[0]), "note": note}


def with_sound(handle, out_path: str, report: dict):
    """The same handle (selection, retime, crop, scale untouched) reading its frames and sound from ``out_path``."""
    from mmx_vendor.c2c_core.video.probe import probe_file

    nh = probe_file(out_path)
    if int(nh.frame_count) != int(handle.frame_count):
        raise ValueError(f"the soundtrack copy has {nh.frame_count} frames, the source {handle.frame_count} - the "
                         "video stream did not copy cleanly; replace the sound in an editor instead.")
    if nh.audio is None:
        raise ValueError("the soundtrack copy has no audio stream.")
    return dataclasses.replace(handle, source=nh.source, stream_index=nh.stream_index, time_base=nh.time_base,
                               audio=nh.audio)
