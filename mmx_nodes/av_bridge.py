# PORTED FROM: https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef@361624f h3_masked_bridge.py
# (MiniMaxH3MaskedAVBridge) and h3_workflow_utilities.py (MiniMaxH3AssembleBridgeAudio), GPL-3.0.
"""H3 AV Bridge / H3 AV Bridge Assemble - generate the middle between two existing clips (an interior insert).

Bridge: the end of clip A and the start of clip B go into the target latent, protected; the sampler generates only
the middle, so it starts exactly where A ends and lands exactly where B begins, in picture AND sound.
Assemble: clip A + the generated middle + clip B, frames and sound on absolute sample boundaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_nodes.av_noise_mask import _streams  # noqa: E402
from mmx_utils.h3_bridge import (  # noqa: E402
    AUDIO_HZ,
    FPS,
    BridgeError,
    assemble,
    canonical_audio,
    check_timing,
    encode_audio_window,
    encode_video_window,
    pixel_frames,
    to_h3_rate,
)

CATEGORY = "MiniMax H3/Long"
PRESERVE_TIP = ("Frames kept at each end: a shared video / audio boundary, 39, 90, 141 ... (39 + 51k). More context "
                "= a smoother join, a shorter generated middle.")


def _nested(parts):
    import comfy.nested_tensor  # noqa: WPS433
    return comfy.nested_tensor.NestedTensor(tuple(parts))


def prepare_bridge(latent, vae, audio_vae, start_frames, start_audio, end_frames, end_audio, preserve_frames=39,
                   crop="center", start_fps=FPS, end_fps=FPS):
    import torch

    # C2C L9.39: the clips' own frame rates (upstream's fps inputs, dropped in the first port) - each clip is
    # resampled to H3's 24 fps on the constant-frame-rate grid; at 24 nothing changes.
    start_frames = to_h3_rate(start_frames, start_fps)
    end_frames = to_h3_rate(end_frames, end_fps)

    v, a = _streams(latent, "H3 AV Bridge")
    if int(v.shape[0]) != 1 or int(a.shape[0]) != 1:
        raise BridgeError("the bridge works on one clip at a time (batch size 1)")
    target = pixel_frames(int(v.shape[2]))
    n = int(preserve_frames)
    check_timing(target, n)
    want_ticks = int(round(target / FPS * AUDIO_HZ))
    if int(a.shape[-1]) != want_ticks:
        raise BridgeError(f"the target latent has {int(a.shape[-1])} audio ticks for {target} frames; expected "
                          f"{want_ticks}. Make it with Empty MiniMax H3 Latent AV.")
    for name, f in (("start_frames", start_frames), ("end_frames", end_frames)):
        if getattr(f, "ndim", 0) != 4 or int(f.shape[0]) < n:
            raise BridgeError(f"{name} must be at least {n} frames [frames, H, W, C] at 24 fps")
    width, height = int(v.shape[4]) * 16, int(v.shape[3]) * 16
    sv, sk = encode_video_window(vae, start_frames[-n:], width, height, crop, n, "clip A's end")
    ev, ek = encode_video_window(vae, end_frames[:n], width, height, crop, n, "clip B's start")
    if (int(v.shape[2]) - ek) % 5:
        raise BridgeError(f"clip B's window would start at latent step {int(v.shape[2]) - ek}, out of H3's 5-step "
                          "phase; check the target length")
    sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    sa, st = encode_audio_window(audio_vae, canonical_audio(start_audio, sr, int(start_frames.shape[0]), "clip A sound"),
                                 n, "tail", "clip A's end sound")
    ea, et = encode_audio_window(audio_vae, canonical_audio(end_audio, sr, int(end_frames.shape[0]), "clip B sound"),
                                 n, "head", "clip B's start sound")
    for name, z, ref in (("clip A video", sv, v), ("clip B video", ev, v)):
        if (z.shape[1], z.shape[3], z.shape[4]) != (ref.shape[1], ref.shape[3], ref.shape[4]):
            raise BridgeError(f"{name} latent {tuple(z.shape)} does not match the target {tuple(ref.shape)}")
    out_v, out_a = v.clone(), a.clone()
    out_v[:, :, :sk] = sv.to(out_v.device, out_v.dtype)
    out_v[:, :, -ek:] = ev.to(out_v.device, out_v.dtype)
    out_a[..., :st] = sa.to(out_a.device, out_a.dtype)
    out_a[..., -et:] = ea.to(out_a.device, out_a.dtype)
    vm = torch.ones((1, 1) + tuple(out_v.shape[2:]), device=out_v.device)
    am = torch.ones((1, 1, int(out_a.shape[2]), int(out_a.shape[3])), device=out_a.device)
    vm[:, :, :sk] = 0.0
    vm[:, :, -ek:] = 0.0
    am[..., :st] = 0.0
    am[..., -et:] = 0.0
    out = dict(latent)
    out["samples"] = _nested((out_v, out_a))
    out["noise_mask"] = _nested((vm, am))
    middle = target - 2 * n
    info = {"target_frames": target, "preserve_frames": n, "middle_frames": middle, "crop": crop,
            "start_fps": float(start_fps), "end_fps": float(end_fps)}
    rates = "" if (abs(float(start_fps) - FPS) < 1e-6 and abs(float(end_fps) - FPS) < 1e-6) else (
        f" (clips converted to 24 fps from {float(start_fps):g} / {float(end_fps):g} fps)")
    report = (f"bridge {target} frames at {width}x{height}{rates}: keep {n} frames of clip A's end + {n} of clip B's start "
              f"({sk} video steps / {st} audio ticks each), generate the middle {middle} frames ({middle / FPS:.2f} s)")
    return out, json.dumps(info), middle, report


class MiniMaxH3_AVBridge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AVBridge",
            display_name="H3 AV Bridge (generate between two clips)",
            category=CATEGORY,
            description=("Generate the middle between two existing clips: the end of clip A and the start of clip B "
                         "are encoded into the target and protected (picture and sound), only the middle is "
                         "generated - so it starts exactly where A ends and lands exactly where B begins. Feed the "
                         "latent to the sampler with your conditioning, decode video and audio, then H3 AV Bridge "
                         "Assemble. 24 fps clips."),
            inputs=[
                io.Latent.Input("latent", tooltip="The target: Empty MiniMax H3 Latent AV, longer than both contexts "
                                                  "(e.g. 192 frames with 39 kept at each end)."),
                io.Vae.Input("vae", tooltip="The H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="The H3 audio VAE."),
                io.Image.Input("start_frames", tooltip="Clip A (or its end), 24 fps."),
                io.Image.Input("end_frames", tooltip="Clip B (or its start), 24 fps."),
                io.Int.Input("preserve_frames", default=39, min=39, max=9984, step=51, tooltip=PRESERVE_TIP),
                io.Combo.Input("crop", options=["center", "disabled"], default="center",
                               tooltip="How the clips are fitted to the target canvas."),
                io.Audio.Input("start_audio", optional=True, tooltip="Clip A's sound (silence if not connected)."),
                io.Audio.Input("end_audio", optional=True, tooltip="Clip B's sound (silence if not connected)."),
                # C2C L9.39: appended (saved widgets keep their positions)
                io.Float.Input("start_fps", default=24.0, min=1.0, max=240.0, step=0.001, optional=True,
                               tooltip="Clip A's frame rate. Anything but 24 is converted to H3's 24 fps on the "
                                       "constant-frame-rate grid (frames repeated or dropped, never blended); its "
                                       "sound keeps its own timing. Assemble converts the same way."),
                io.Float.Input("end_fps", default=24.0, min=1.0, max=240.0, step=0.001, optional=True,
                               tooltip="Clip B's frame rate, converted the same way."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="bridge_info", tooltip="For H3 AV Bridge Assemble."),
                io.Int.Output(display_name="middle_frames"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, audio_vae, start_frames, end_frames, preserve_frames=39, crop="center",
                start_audio=None, end_audio=None, start_fps=24.0, end_fps=24.0) -> io.NodeOutput:
        try:
            out, info, middle, report = prepare_bridge(latent, vae, audio_vae, start_frames, start_audio, end_frames,
                                                       end_audio, preserve_frames, crop, start_fps, end_fps)
        except BridgeError as exc:
            raise ValueError(f"H3 AV Bridge: {exc}") from exc
        return io.NodeOutput(out, info, middle, report, ui={"text": [report]})


class MiniMaxH3_AVBridgeAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_AVBridgeAssemble",
            display_name="H3 AV Bridge Assemble",
            category=CATEGORY,
            description=("Clip A + the generated middle + clip B, picture and sound: the bridge's two kept contexts "
                         "are dropped (the clips themselves are used there) and the sound is cut on the picture's "
                         "absolute sample boundaries, so nothing drifts."),
            inputs=[
                io.Image.Input("images", tooltip="The decoded bridge (all of its frames)."),
                io.Audio.Input("audio", tooltip="The decoded bridge sound."),
                io.String.Input("bridge_info", force_input=True, tooltip="From H3 AV Bridge."),
                io.Image.Input("start_frames", tooltip="Clip A, as given to the bridge."),
                io.Image.Input("end_frames", tooltip="Clip B, as given to the bridge."),
                io.Audio.Input("start_audio", optional=True),
                io.Audio.Input("end_audio", optional=True),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, images, audio, bridge_info, start_frames, end_frames, start_audio=None,
                end_audio=None) -> io.NodeOutput:
        try:
            info = json.loads(bridge_info)
            # the same 24 fps conversion the bridge used (older bridge_info has none: 24)
            start_frames = to_h3_rate(start_frames, float(info.get("start_fps", 24.0)))
            end_frames = to_h3_rate(end_frames, float(info.get("end_fps", 24.0)))
            frames, sound, report = assemble(images, audio, start_frames, start_audio, end_frames, end_audio,
                                             int(info["target_frames"]), int(info["preserve_frames"]),
                                             info.get("crop", "center"))
        except (BridgeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"H3 AV Bridge Assemble: {exc}") from exc
        return io.NodeOutput(frames, sound, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_AVBridge", "MiniMaxH3_AVBridgeAssemble", "prepare_bridge"]
