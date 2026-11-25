# PORTED FROM: https://github.com/slmonker/ComfyUI-OpenH3Regenerate2KIC@a85bd96 nodes.py
#              (OpenH3ICDraft, OpenH3ICPrompt, OpenH3ICSigmas, OpenH3ICComposeLatent) and tools/convert_ic_lora.py
#              (key layout, applied at load time). Apache-2.0, author slmonker.
# Modified by Code2Collapse, 2026-11: V3 schema, run-time silence option, asset length check, LoRA chain node,
# appended outputs. Distributed as part of a GPL-3.0 work.
"""Open-H3 Regenerate 2K IC (spec docs/research/H3_REGEN_IC_spec.md): five V3 nodes.

A pipeline distinct from the STD Open Regen nodes: Ref2VA base + three plain LoRAs in a fixed order, core
MiniMaxH3ReferenceToVideo conditioning with the draft as <Video 1> and the verbatim TASK prompt, the up3/up4/hf8
grids, a target latent of the Ref2VA shape with pinned silence. The 2K upscale is MiniMaxH3_LatentUpscaleModel.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import open_regen_ic as R  # noqa: E402
from mmx_utils.av_latent_ops import frame_count_from_video_t  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.open_regen_ic")
CATEGORY = "MiniMax H3/DEPR/Regenerate IC"   # 2026-11-25: replaced by H3 Open Regen (open_regen_all.py)
ASSET_FOLDER = "h3_assets"
SILENCE_ENCODE, SILENCE_ASSET = "encode (audio VAE)", "asset file"
NO_FILE = "(none)"
NO_LORAS = "<no loras found>"
# 2026-11-25 (owner): the IC recipe now runs on the H3 Open Regen nodes (mmx_nodes/open_regen.py IC options).
SUPERSEDED = (" Superseded by the H3 Open Regen nodes (IC options: Prepare Draft fps_map floor, LoRA speed/sharp "
              "slots, Sigmas IC grids, Refine Latent reference_latent); kept so saved workflows load.")

try:
    import folder_paths
except Exception:  # noqa: BLE001 - not just ImportError: a broken core must not unregister the pack
    folder_paths = None  # type: ignore[assignment]

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path(ASSET_FOLDER, os.path.join(folder_paths.models_dir, ASSET_FOLDER))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not register models/%s: %s", ASSET_FOLDER, exc)


def _names(folder: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list(folder))
    except Exception:  # noqa: BLE001
        return []


def _asset_names() -> list[str]:
    return [n for n in _names(ASSET_FOLDER) if n.lower().endswith(".safetensors")]


def _full_path(folder: str, name: str, what: str) -> str:
    path = folder_paths.get_full_path(folder, name) if (folder_paths is not None and name) else None
    if not path:
        raise RuntimeError(f"{what}: {name!r} was not found in models/{folder}. Put the file there (or pick one "
                           "from the list) and run again.")
    return path


def _load_safetensors(path: str, what: str) -> dict:
    """safe_open on CPU; each tensor cloned off the file so nothing stays memory-mapped (Windows error 1224)."""
    if not path.lower().endswith(".safetensors"):
        raise RuntimeError(f"{what}: only .safetensors files are loaded (pickle formats are refused): {path}")
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k).clone() for k in f.keys()}


def _nested(parts):
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor(tuple(parts))


# --------------------------------------------------------------------------------------------- Prepare Draft

class MiniMaxH3_OpenRegenICPrepareDraft(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenICPrepareDraft",
            display_name="DEPR · H3 Open Regen IC · Prepare Draft (17k+5 / 24 fps)",
            category=CATEGORY,
            description=("IC draft preparation (port of OpenH3ICDraft): resample to 24 fps with FLOOR (STD uses round), "
                         "repeat the last frame up to frame_count, cover or stretch to the draft canvas, fit the "
                         "soundtrack. Needs at least 113 real frames (~4.7 s)." + SUPERSEDED),
            inputs=[
                io.Image.Input("images", tooltip="Draft frames [N,H,W,C]; alpha is dropped."),
                io.Float.Input("source_fps", default=24.0, min=1.0, max=240.0, step=0.01,
                               tooltip="Wire GetVideoComponents.fps. Index map int(i*fps/24) = floor (STD rounds)."),
                io.Int.Input("target_width", default=1344, min=32, max=8192, step=32,
                             tooltip="Draft canvas width in px; must be a multiple of 32 (checked). Frames are "
                                     "covered / stretched to it (see fit)."),
                io.Int.Input("target_height", default=768, min=32, max=8192, step=32,
                             tooltip="Draft canvas height in px; must be a multiple of 32 (checked). Frames "
                                     "are covered / stretched to it (see fit)."),
                io.Int.Input("frame_count", default=124, min=124, max=362, step=17,
                             tooltip="17k+5. 124 is the tested length; above 124 needs Compose silence_source = "
                                     "encode (audio VAE). Needs >= 113 real frames from the source."),
                io.Int.Input("output_fps", default=24, min=1, max=120,
                             tooltip="Kept for parity with upstream: must be 24."),
                io.Combo.Input("fit", options=["cover", "stretch"], default="cover",
                               tooltip="cover = scale to cover + centre crop; stretch distorts a different aspect."),
                io.Combo.Input("audio_mode", options=["trim_pad", "passthrough", "silent"], default="trim_pad",
                               tooltip="trim_pad: cut / zero-pad to frame_count/24 s at the source rate. "
                                       "passthrough: unchanged (CreateVideo may get a longer track). "
                                       "silent (or no audio): zeros at 32 kHz."),
                io.Audio.Input("audio", optional=True, tooltip="Draft soundtrack."),
            ],
            outputs=[
                io.Image.Output("draft_frames", tooltip="IMAGE batch of exactly frame_count frames (17k+5) at 24 "
                                                        "fps on the target_width x target_height canvas; the "
                                                        "last source frame repeats past the source's end."),
                io.Audio.Output("draft_audio", tooltip="Soundtrack per audio_mode: cut / zero-padded to "
                                                       "frame_count / 24 s, unchanged (passthrough), or "
                                                       "silence at 32 kHz."),
                io.String.Output("report", tooltip="JSON: source frames and fps, real frames at 24 fps, output "
                                                   "frames, canvas, audio samples / rate, fit and audio mode."),
                io.Int.Output("frame_count", tooltip="Frames in draft_frames (= the frame_count input, 17k+5)."),
                io.Int.Output("real_frames", tooltip="Source frames inside the clip before the last one repeats "
                                                     "(feed STD Finish to drop the padded tail)."),
            ],
        )

    @classmethod
    def execute(cls, images, source_fps, target_width, target_height, frame_count, output_fps,
                fit="cover", audio_mode="trim_pad", audio=None) -> io.NodeOutput:
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("OpenH3ICDraft expects IMAGE [T,H,W,C]")
        n = int(images.shape[0])
        R.validate_draft(n, target_width, target_height, frame_count, output_fps)
        indices, real = R.floor_index_map(n, float(source_fps), int(frame_count), int(output_fps))
        picked = images.index_select(0, torch.tensor(indices, device=images.device, dtype=torch.long))
        if fit == "cover":
            frames = R.cover_resize(picked, int(target_height), int(target_width))
        elif fit == "stretch":
            frames = R.stretch_resize(picked, int(target_height), int(target_width))
        else:
            raise ValueError(f"Unknown fit {fit!r}")
        out_audio = R.prepare_audio(audio_mode, audio, int(frames.shape[0]), int(output_fps))
        report = {
            "source_frames": n,
            "source_fps": float(source_fps),
            "real_frames_at_24fps": real,
            "output_frames": int(frames.shape[0]),
            "output_fps": int(output_fps),
            "canvas": [int(target_width), int(target_height)],
            "audio_samples": int(out_audio["waveform"].shape[-1]),
            "audio_rate": int(out_audio["sample_rate"]),
            "fps_map": "floor",
            "fit": fit,
            "audio_mode": audio_mode if audio is not None else "silent (no audio connected)",
        }
        return io.NodeOutput(frames, out_audio, json.dumps(report, ensure_ascii=False), int(frames.shape[0]), real)


# --------------------------------------------------------------------------------------------- Prompt

class MiniMaxH3_OpenRegenICPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenICPrompt",
            display_name="DEPR · H3 Open Regen IC · Official Prompt",
            category=CATEGORY,
            description="The verbatim IC TASK prompt + 'Scene description: <caption>'. Feed the IC refine Ref2VA.",
            inputs=[io.String.Input("caption", multiline=True, default="",
                                    tooltip="Scene description appended (trimmed) after the fixed IC TASK prompt "
                                            "and 'Scene description: '. Empty = task prompt only.")],
            outputs=[io.String.Output("prompt", tooltip="The verbatim IC TASK prompt + 'Scene description: ' + "
                                                        "caption. Wire to the Ref2VA (Reference to Video) "
                                                        "prompt.")],
        )

    @classmethod
    def execute(cls, caption="") -> io.NodeOutput:
        return io.NodeOutput(R.build_prompt(caption))


# --------------------------------------------------------------------------------------------- Sigmas

class MiniMaxH3_OpenRegenICSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenICSigmas",
            display_name="DEPR · H3 Open Regen IC · Sigma Grid",
            category=CATEGORY,
            description="Fixed IC grids: up3 (3 steps, default), up4, hf8. Use euler + BasicGuider CFG 1." + SUPERSEDED,
            inputs=[
                io.Combo.Input("sampler", options=list(R.SIGMAS), default="up3",
                               tooltip="Fixed IC sigma grid. up3: 3 steps from 0.835; up4: 4 steps from 0.923; "
                                       "hf8: 8 steps from 1.0. Ignored when custom_sigmas is set."),
                io.String.Input("custom_sigmas", default="", multiline=False,
                                tooltip="Comma list; overrides sampler. Must end at 0, never increase, stay in [0,1]."),
            ],
            outputs=[io.Sigmas.Output("sigmas", tooltip="SIGMAS of the chosen grid (or custom list), ending at "
                                                        "0. Use with euler + BasicGuider CFG 1."),
                     io.String.Output("report", tooltip="JSON: which grid / custom list was used and its "
                                                        "values.")],
        )

    @classmethod
    def execute(cls, sampler="up3", custom_sigmas="") -> io.NodeOutput:
        values, report = R.build_sigmas(sampler, custom_sigmas)
        return io.NodeOutput(torch.tensor(values, dtype=torch.float32), json.dumps(report))


# --------------------------------------------------------------------------------------------- Compose

def _encode_silence(audio_vae, frames: int) -> torch.Tensor:
    """Digital silence of ``frames``/24 s encoded the way core VAEEncodeAudio does (no resample needed: the
    waveform is built at the VAE's own rate)."""
    sr = getattr(audio_vae, "audio_sample_rate", None)
    if not sr:
        raise ValueError("audio_vae has no audio_sample_rate: connect the MiniMax H3 audio VAE")
    waveform = torch.zeros((1, 2, R.silence_sample_count(frames, int(sr))), dtype=torch.float32)
    latent = audio_vae.encode(waveform.movedim(1, -1))
    if isinstance(latent, dict):
        latent = latent.get("samples")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or tuple(latent.shape[1:3]) != (32, 2):
        shape = tuple(latent.shape) if isinstance(latent, torch.Tensor) else type(latent).__name__
        raise ValueError(f"audio_vae returned {shape}, expected [B, 32, 2, T]: connect the MiniMax H3 audio VAE")
    return latent


class MiniMaxH3_OpenRegenICComposeLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        assets = _asset_names()
        default_asset = R.pick_default(assets, ("silence_audio_124f",), assets[0] if assets else NO_FILE)
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenICComposeLatent",
            display_name="DEPR · H3 Open Regen IC · Compose Target Latent",
            category=CATEGORY,
            description=("Target AV latent for the IC refine: the upscaled draft video (must equal the Ref2VA "
                         "width/height/length, 2560x1440x124 by default) + silence pinned by noise mask 0." + SUPERSEDED),
            inputs=[
                io.Latent.Input("video_latent", tooltip="The upscaled draft (video or AV latent; video part used)."),
                io.Latent.Input("reference_latent",
                                tooltip="LATENT output of core MiniMax H3 Reference to Video (two-part AV latent)."),
                io.Boolean.Input("pin_audio", default=True,
                                 tooltip="Noise mask 1 on video, 0 on audio. Off removes the noise mask."),
                io.Combo.Input("silence_source", options=[SILENCE_ENCODE, SILENCE_ASSET], default=SILENCE_ENCODE,
                               tooltip="encode: digital silence encoded at run time with the audio VAE (any length). "
                                       "asset file: a user-placed silence_audio_124f.safetensors in models/h3_assets "
                                       "(124 frames only; never shipped with this pack)."),
                io.Vae.Input("audio_vae", optional=True, tooltip="MiniMax H3 audio VAE; required for encode."),
                io.Combo.Input("silence_file", options=assets or [NO_FILE], default=default_asset, optional=True,
                               tooltip="File in models/h3_assets for silence_source = asset file."),
            ],
            outputs=[io.Latent.Output("latent", tooltip="Two-part AV latent: the upscaled video + silence, "
                                                        "keeping reference_latent's other keys; with pin_audio "
                                                        "a noise mask of 1 on video, 0 on audio. Wire to the "
                                                        "refine sampler."),
                     io.String.Output("report", tooltip="JSON: video and audio latent shapes, frames, silence "
                                                        "source, pinned.")],
        )

    @classmethod
    def execute(cls, video_latent, reference_latent, pin_audio=True, silence_source=SILENCE_ENCODE,
                audio_vae=None, silence_file=None) -> io.NodeOutput:
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise ValueError("video_latent must be a LATENT")
        video = video_latent["samples"]
        if getattr(video, "is_nested", False):
            video = video.tensors[0]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("video_latent must hold an H3 video latent [B, 24, T, h, w]")
        ref = reference_latent.get("samples") if isinstance(reference_latent, dict) else None
        if not getattr(ref, "is_nested", False) or len(ref.tensors) != 2:
            raise ValueError("reference_latent must be the joint AV latent from MiniMax H3 Reference to Video")
        ref_video, ref_audio = ref.tensors
        video = video.to(dtype=ref_video.dtype, device=ref_video.device)
        if tuple(video.shape) != tuple(ref_video.shape):
            raise ValueError(
                "Upscaled video latent shape %s does not match reference AV target shape %s. Make the Reference "
                "to Video width / height / length equal the upscale target and the Prepare Draft frame_count."
                % (tuple(video.shape), tuple(ref_video.shape)))
        frames = frame_count_from_video_t(int(ref_video.shape[2]))

        if silence_source == SILENCE_ENCODE:
            if audio_vae is None:
                raise ValueError("silence_source = encode needs the MiniMax H3 audio VAE on audio_vae")
            silence = _encode_silence(audio_vae, frames)
            source = "encoded silence (audio VAE, run time)"
        elif silence_source == SILENCE_ASSET:
            if ref_audio.shape[-1] != 207 or frames != R.ASSET_FRAMES:
                raise ValueError(f"The silence asset encodes {R.ASSET_FRAMES} frames only; this target has {frames} "
                                 f"frames ({ref_audio.shape[-1]} audio latents). Use silence_source = encode.")
            if not silence_file or silence_file == NO_FILE:
                raise ValueError("silence_source = asset file needs silence_audio_124f.safetensors in models/h3_assets")
            path = _full_path(ASSET_FOLDER, silence_file, "Compose Target Latent")
            sd = _load_safetensors(path, "Compose Target Latent")
            if "audio_latents" not in sd:
                raise ValueError(f"{silence_file!r} has no 'audio_latents' tensor: not the IC silence asset")
            silence = R.asset_to_latent(sd["audio_latents"])
            if silence.shape[-1] != ref_audio.shape[-1]:
                raise ValueError(f"silence asset has {silence.shape[-1]} audio latents, the target needs "
                                 f"{ref_audio.shape[-1]}")
            source = f"asset file {silence_file}"
        else:
            raise ValueError(f"Unknown silence_source {silence_source!r}")
        silence = R.fit_audio_columns(silence.to(device=ref_audio.device, dtype=ref_audio.dtype), ref_audio)

        out = R.pinned_av_latent(reference_latent, video_latent, video, silence, bool(pin_audio), _nested,
                                 "open_h3_ic", {"audio": source, "pinned": bool(pin_audio)})
        report = {"video": list(video.shape), "audio": list(silence.shape), "frames": frames,
                  "silence_source": source, "pinned": bool(pin_audio)}
        return io.NodeOutput(out, json.dumps(report))


# --------------------------------------------------------------------------------------------- LoRA Chain

def _matched(before, after) -> int:
    old = getattr(before, "patches", {}) or {}
    new = getattr(after, "patches", {}) or {}
    return sum(1 for k, v in new.items() if len(v) > len(old.get(k, ())))


def _apply_comfy(model, sd, strength):
    import comfy.sd
    patched, _ = comfy.sd.load_lora_for_models(model, None, sd, strength, 0)
    return patched, _matched(model, patched)


def _apply_diffusers(model, sd, strength):
    import comfy.lora
    lora, to_load, expect = R.diffusers_plan(sd)
    shapes = {k: tuple(v.shape) for k, v in model.model.state_dict().items()}
    R.check_targets(expect, shapes)
    patches = comfy.lora.load_lora(lora, to_load)
    if len(patches) != len(to_load):
        raise ValueError(f"core loaded {len(patches)} of {len(to_load)} IC LoRA modules")
    patched = model.clone()
    applied = patched.add_patches(patches, strength)
    if len(applied) != len(to_load):
        raise ValueError(f"only {len(applied)} of {len(to_load)} IC LoRA patches matched this model")
    return patched, len(applied)


class MiniMaxH3_OpenRegenICLoRAChain(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        loras = _names("loras")
        speed = R.pick_default(loras, ("hyperflow_8step",))
        sharp = R.pick_default(loras, ("minimax_h3_lms",))
        ic_opts = loras or [NO_LORAS]
        ic = R.pick_default(loras, ("open_regenerate_2k_ic", "open_h3_regenerate_2k_ic"), ic_opts[0])
        opt = ["none"] + loras
        return io.Schema(
            node_id="MiniMaxH3_OpenRegenICLoRAChain",
            display_name="DEPR · H3 Open Regen IC · LoRA Chain (speed → LMS Sharp → IC)",
            category=CATEGORY,
            description=("The IC refine adapters in the documented order, fixed in code: speed (HyperFlow 8-step; "
                         "DMAD 4-step accepted) -> LMS Sharp r64 -> Open-H3 IC. Base = minimax_h3_ref2va_bf16 or "
                         "the hybrid FL2VA/Ref2VA." + SUPERSEDED),
            inputs=[
                io.Model.Input("model", tooltip="Base H3 model: minimax_h3_ref2va_bf16 or the hybrid "
                                                "FL2VA/Ref2VA. The three LoRAs are patched onto it in order."),
                io.Combo.Input("speed_lora", options=opt, default=speed,
                               tooltip="Slot 1, applied first: the speed LoRA (HyperFlow 8-step; DMAD 4-step "
                                       "accepted). none = skipped (a warning goes to the report)."),
                io.Float.Input("speed_strength", default=1.0, min=-2.0, max=2.0, step=0.01,
                               tooltip="Model strength of the speed LoRA; 0 adds no change."),
                io.Combo.Input("sharp_lora", options=opt, default=sharp,
                               tooltip="Slot 2, applied second: LMS Sharp r64. none = skipped (a warning goes "
                                       "to the report)."),
                io.Float.Input("sharp_strength", default=1.0, min=-2.0, max=2.0, step=0.01,
                               tooltip="Model strength of the LMS Sharp LoRA; 0 adds no change."),
                io.Combo.Input("ic_lora", options=ic_opts, default=ic,
                               tooltip="ComfyUI-keyed IC LoRA, or the official Diffusers-keyed file (converted "
                                       "at load)."),
                io.Float.Input("ic_strength", default=1.0, min=-2.0, max=2.0, step=0.01,
                               tooltip="Model strength of the IC LoRA (applied last)."),
                io.Combo.Input("ic_format", options=["auto", "comfy", "diffusers (convert at load)"], default="auto",
                               advanced=True, tooltip="auto = detect by key names."),
            ],
            outputs=[io.Model.Output("model", tooltip="Clone of the model with speed -> LMS Sharp -> IC LoRA "
                                                      "patches. Wire to the IC refine sampler."),
                     io.String.Output("report", tooltip="Text: slot-order warnings, then per slot the file, "
                                                        "strength, key format and modules matched.")],
        )

    @classmethod
    def execute(cls, model, speed_lora="none", speed_strength=1.0, sharp_lora="none", sharp_strength=1.0,
                ic_lora=NO_LORAS, ic_strength=1.0, ic_format="auto") -> io.NodeOutput:
        chosen = {"speed": speed_lora, "sharp": sharp_lora, "ic": ic_lora}
        strengths = {"speed": float(speed_strength), "sharp": float(sharp_strength), "ic": float(ic_strength)}
        if ic_lora in (None, "", "none", NO_LORAS):
            raise ValueError("ic_lora is required: put the Open-H3 Regenerate 2K IC LoRA in models/loras")
        lines = R.order_warnings(chosen)
        for slot in R.SLOTS:
            name = chosen[slot]
            if name in (None, "", "none"):
                lines.append(f"{slot}: none (skipped)")
                continue
            sd = _load_safetensors(_full_path("loras", name, f"IC LoRA Chain ({slot})"), f"IC LoRA Chain ({slot})")
            if slot == "ic" and ic_format != "auto":
                fmt = "diffusers" if ic_format.startswith("diffusers") else "comfy"
            else:
                fmt = R.detect_lora_format(sd.keys())
            if fmt == "diffusers":
                model, n = _apply_diffusers(model, sd, strengths[slot])
            else:
                model, n = _apply_comfy(model, sd, strengths[slot])
                if n == 0:
                    raise ValueError(f"{slot} LoRA {name!r}: no key matched this model. Wrong base model (load "
                                     "the MiniMax H3 Ref2VA / hybrid model) or not a ComfyUI-keyed LoRA.")
            del sd
            lines.append(f"{slot}: {name} strength={strengths[slot]} format={fmt} modules_matched={n}")
        return io.NodeOutput(model, "\n".join(lines))


NODES = [
    MiniMaxH3_OpenRegenICPrepareDraft,
    MiniMaxH3_OpenRegenICPrompt,
    MiniMaxH3_OpenRegenICSigmas,
    MiniMaxH3_OpenRegenICComposeLatent,
    MiniMaxH3_OpenRegenICLoRAChain,
]
