# PORTED FROM: MaskVidExperiments (third_party/MaskVidExperiments/nodes_audio_mask.py) by drozbay
# Licence: GPL-3.0 — direct copy authorised by owner; attribution retained.
"""Time-range noise masking for the audio side of AV and audio latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.nested_tensor
from comfy_api.latest import io

CATEGORY = "MiniMax H3/Mask"


def _split_latent(latent):
    samples = latent["samples"]
    video = None
    if samples.is_nested:
        video, audio = samples.unbind()
    elif samples.ndim in (3, 4):
        audio = samples
    else:
        raise ValueError("connect a joint AV latent or an audio latent")
    return video, audio


def _resolve_timing(timing, vae, audio):
    if timing["timing"] == "auto":
        if vae is None:
            raise ValueError("auto timing requires a VAE to be connected")
        inner = getattr(vae, "first_stage_model", None)
        lps = getattr(inner, "latents_per_second", None)
        if lps is None:
            rate = getattr(vae, "audio_sample_rate", None)
            down = getattr(vae, "downscale_ratio", None)
            if rate is None or not isinstance(down, (int, float)):
                raise ValueError("the connected VAE does not expose audio timing, connect the model's audio VAE or use manual timing")
            lps = rate / down
        spectrogram = getattr(inner, "mel_bins", None) is not None
    else:
        lps = timing["latents_per_second"]
        spectrogram = timing["layout"] == "time then bins"
    axis = audio.ndim - 2 if spectrogram and audio.ndim >= 4 else audio.ndim - 1
    return float(lps), axis


def _timing_input():
    return io.DynamicCombo.Input(
        "timing",
        tooltip="auto: read the audio latent rate and layout from the connected audio VAE. manual: enter them directly.",
        options=[
            io.DynamicCombo.Option("auto", []),
            io.DynamicCombo.Option("manual", [
                io.Float.Input("latents_per_second", default=40.0, min=0.01, max=1000.0, step=0.01,
                               tooltip="Audio latent frames per second. 40 for MiniMax H3, 25 for LTX."),
                io.Combo.Input("layout", options=["time last", "time then bins"], default="time last",
                               tooltip="Where time runs in the audio latent. time last for MiniMax H3 and audio-only models, time then bins for LTX spectrogram latents."),
            ]),
        ],
    )


def _parse_time_ranges(text):
    values = [float(v) for v in text.split(",") if v.strip()]
    if len(values) % 2:
        raise ValueError("time_ranges needs an even count of values (in,out pairs)")
    pairs = list(zip(values[::2], values[1::2]))
    for s, e in pairs:
        if e <= s:
            raise ValueError(f"time_ranges pair {s:g},{e:g}: out must be after in")
    return pairs


def _fill_range(mask, axis, start, end):
    idx = [slice(None)] * mask.ndim
    idx[axis] = slice(start, end)
    mask[tuple(idx)] = 1.0


def _timeline_curve(mask, t):
    flat = mask.reshape(-1, mask.shape[-1]).amax(dim=0)
    return F.adaptive_max_pool1d(flat.float()[None, None], t)[0, 0]


class MiniMaxH3_AudioMaskToLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_AudioMaskToLatent",
            display_name="H3 Audio Mask To Latent",
            category=CATEGORY,
            description="Regenerate the selected audio time ranges and keep the rest. Ranges come from time_ranges when set, else from the mask when connected, else from start_time/end_time. Accepts a joint AV latent or an audio latent alone. On a joint latent without an existing noise mask the video is fully preserved; apply Set Latent Noise Mask before this node to control the video side.",
            inputs=[
                io.Latent.Input("latent", tooltip="Joint AV latent or audio latent to mask."),
                io.Vae.Input("vae", optional=True,
                             tooltip="The audio VAE used to encode this latent (for a joint AV latent, the audio VAE, not the video one). Required when timing is auto."),
                _timing_input(),
                io.Float.Input("start_time", default=0.0, min=0.0, max=10000.0, step=0.01,
                               tooltip="Start of the regenerated audio range in seconds. Ignored when time_ranges or mask is used."),
                io.Float.Input("end_time", default=1.0, min=0.0, max=10000.0, step=0.01,
                               tooltip="End of the regenerated audio range in seconds. Ignored when time_ranges or mask is used."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Mask over the audio timeline: time runs along the width axis, stretched to the audio duration. White = regenerate, black = keep."),
                io.String.Input("time_ranges", optional=True, default="",
                                tooltip="Comma-separated seconds forming in,out pairs (e.g. 0,1.5,3,4.25). Takes priority over mask and start/end when set."),
                io.Combo.Input("existing_mask", options=["keep", "replace"], default="keep",
                               tooltip="keep: merge the new ranges into the audio mask already on the latent, so copies of this node chain. replace: discard the existing audio mask first. The video side of a joint mask is kept either way."),
            ],
            outputs=[
                io.Latent.Output(tooltip="The latent with the audio noise mask attached."),
                io.String.Output(display_name="report", tooltip="Which audio time ranges were marked for regeneration."),
            ],
        )

    @classmethod
    def execute(cls, latent, timing, start_time, end_time, existing_mask, mask=None, time_ranges="", vae=None) -> io.NodeOutput:
        video, audio = _split_latent(latent)
        lps, axis = _resolve_timing(timing, vae, audio)
        t = audio.shape[axis]

        video_mask = torch.zeros_like(video[:, :1]) if video is not None else None
        audio_mask = None
        prior = latent.get("noise_mask", None)
        if prior is not None:
            if prior.is_nested:
                masks = prior.unbind()
                video_mask = masks[0]
                audio_mask = masks[1]
            elif video is None:
                audio_mask = prior
            else:
                video_mask = prior
        if existing_mask == "replace" or audio_mask is None:
            audio_mask = torch.zeros_like(audio[:, :1], dtype=torch.float32)
        else:
            audio_mask = audio_mask.clone().float()

        pairs = _parse_time_ranges(time_ranges)
        source = "time_ranges"
        if pairs:
            for s, e in pairs:
                _fill_range(audio_mask, axis, max(0, round(s * lps)), min(t, round(e * lps)))
        elif mask is not None:
            source = "timeline mask"
            shape = [1] * audio_mask.ndim
            shape[axis] = t
            curve = _timeline_curve(mask, t).to(audio_mask)
            audio_mask = torch.maximum(audio_mask, curve.view(shape))
        else:
            if end_time <= start_time:
                raise ValueError("end_time must be after start_time")
            source = f"start/end ({start_time:.2f}s–{end_time:.2f}s)"
            _fill_range(audio_mask, axis, max(0, round(start_time * lps)), min(t, round(end_time * lps)))

        out = latent.copy()
        if video is None:
            out["noise_mask"] = audio_mask
        else:
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        filled = int(audio_mask.reshape(-1, t).amax(dim=0).sum().item()) if t else 0
        report = (f"source: {source}\nlatent_frames: {t} at {lps:g}/s\n"
                  f"frames_marked_generate: {filled}/{t}")
        return io.NodeOutput(out, report)


def _mask_runs(vals, lps):
    parts = []
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            v = vals[start]
            label = "keep" if v == 0.0 else "generate" if v == 1.0 else "soft"
            parts.append(f"{start / lps:.2f}-{i / lps:.2f}s = {v:g} ({label})")
            start = i
    return ", ".join(parts)


class MiniMaxH3_AudioMaskDebug(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_AudioMaskDebug",
            display_name="H3 Audio Mask Debug",
            category=CATEGORY,
            description="Report which audio time ranges the latent's noise mask keeps (0.0) and generates (1.0). Accepts a joint AV latent or an audio latent alone; route the string to a Preview Any node.",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae", optional=True,
                             tooltip="The audio VAE used to encode this latent. Required when timing is auto."),
                _timing_input(),
            ],
            outputs=[io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, latent, timing, vae=None) -> io.NodeOutput:
        video, audio = _split_latent(latent)
        lps, axis = _resolve_timing(timing, vae, audio)
        t = audio.shape[axis]
        dur = t / lps

        prior = latent.get("noise_mask", None)
        if prior is None:
            return io.NodeOutput(f"no noise mask on the latent: all 0.00-{dur:.2f}s generate (full denoise)")
        if video is not None:
            if prior.is_nested:
                amask = prior.unbind()[1]
                source = "audio side of nested mask"
            else:
                return io.NodeOutput(f"plain (video-only) mask on joint latent: audio side missing, the sampler pads it with ones, all 0.00-{dur:.2f}s generate")
        else:
            amask = prior
            source = "mask on audio latent"

        m = amask.float().movedim(axis, -1).reshape(-1, t)
        lines = [f"{source}: {t} latent frames, {dur:.2f}s at {lps:g}/s"]
        if not torch.equal(m.amax(dim=0), m.amin(dim=0)):
            lines.append("warning: values vary within frames, reporting the per-frame max")
        vals = [round(float(v), 3) for v in m.amax(dim=0)]
        lines.append(_mask_runs(vals, lps))
        return io.NodeOutput("\n".join(lines))
