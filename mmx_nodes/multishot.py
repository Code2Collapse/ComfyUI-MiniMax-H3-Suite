"""H3 multishot script split and chained samplers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.multishot import (  # noqa: E402
    frames_per_shot_from_seconds,
    normalize_shot_list,
    parse_script,
    seam_trim_samples,
    shot_count_for_audio,
    split_script_for_workflow,
    trim_master_to_soundtrack,
    xfade_audio,
)
from mmx_utils.native_audio import lock_audio_chunk_into_latent, resample_waveform  # noqa: E402

CATEGORY = "MiniMax H3/Sampling"


def _nested_factory():
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor


class MiniMaxH3_ScriptSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ScriptSplit",
            display_name="H3 Shot List",
            category=CATEGORY,
            description=(
                "Split a multishot script into per-shot prompt strings. Plain text with --- "
                "separators or JSON {\"prompts\": [...]}."
            ),
            inputs=[
                io.String.Input(
                    "script",
                    multiline=True,
                    default="Shot 1 prompt goes here.\n---\n"
                    "Shot 2 prompt goes here.\n---\n"
                    "Shot 3 prompt goes here.",
                    tooltip="One prompt per shot, separated by --- on its own line.",
                ),
                io.Int.Input(
                    "shot_count",
                    default=0,
                    min=0,
                    max=3,
                    tooltip="0 = count from the script. 2 = third segment continues scene 2.",
                ),
            ],
            outputs=[
                io.String.Output("shot_1"),
                io.String.Output("shot_2"),
                io.String.Output("shot_3"),
                io.String.Output("shot_4"),
                io.Int.Output("shot_count"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(cls, script, shot_count=0) -> io.NodeOutput:
        shots, n, report = split_script_for_workflow(script, shot_count=shot_count)
        return io.NodeOutput(*shots, n, report)


class MiniMaxH3_MultishotSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MultishotSampler",
            display_name="H3 Multishot Sampler (one node)",
            category=CATEGORY,
            description=(
                "Parse a multishot script, sample each shot with frame chaining, seam-trim, "
                "and optional exact-audio lock per shot. JoyEcho-style complexity lives inside "
                "one node so the canvas stays legible."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input(
                    "script",
                    multiline=True,
                    default="Shot 1 prompt goes here.\n---\n"
                    "Shot 2 prompt goes here.\n---\n"
                    "Shot 3 prompt goes here.",
                ),
                io.Int.Input("shot_count", default=0, min=0, max=64),
                io.Int.Input("width", default=768, min=32, max=4096, step=32),
                io.Int.Input("height", default=1344, min=32, max=4096, step=32),
                io.Float.Input("seconds_per_shot", default=5.0, min=0.25, max=15.0, step=0.25),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=50),
                io.Boolean.Input("seed_per_shot", default=True),
                io.Image.Input("start_image", optional=True),
                io.Audio.Input("exact_audio", optional=True),
                io.Image.Input("reference_image", optional=True),
                io.Combo.Input("reference_image_size", options=["match", "max"], default="match"),
                io.String.Input("checkpoint_prefix", default="video/H3_CHAINED_CHECKPOINTS", optional=True),
            ],
            outputs=[
                io.Image.Output("master_frames"),
                io.Audio.Output("master_audio"),
                io.Int.Output("shots_rendered"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        script,
        shot_count,
        width,
        height,
        seconds_per_shot,
        seed,
        steps,
        seed_per_shot=True,
        start_image=None,
        exact_audio=None,
        reference_image=None,
        reference_image_size="match",
        checkpoint_prefix="",
    ) -> io.NodeOutput:
        import node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio

        frames_per_shot = frames_per_shot_from_seconds(seconds_per_shot)
        shots = parse_script(script)
        if exact_audio is not None and shot_count == 0:
            n = shot_count_for_audio(exact_audio, frames_per_shot)
        else:
            n = shot_count if shot_count > 0 else len(shots)
        shots, plan_notes = normalize_shot_list(shots, n)

        sigmas = ncs.BasicScheduler().get_sigmas(model, "simple", steps, 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler("res_multistep")[0]

        frames_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        sr = None
        prev_last = None
        notes = list(plan_notes)

        ref_items, ref_blocks = [], []
        if reference_image is not None:
            ref = reference_image[:1]
            rh, rw = ref.shape[1], ref.shape[2]
            if reference_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (rw * rh)))
            else:
                scale = min(1.0, mmh3.REF_IMAGE_SHORT_EDGE / min(rw, rh))
            tw = max(mmh3.CANVAS_MULTIPLE, round(rw * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE)
            th = max(mmh3.CANVAS_MULTIPLE, round(rh * scale / mmh3.CANVAS_MULTIPLE) * mmh3.CANVAS_MULTIPLE)
            ref_resized = mmh3._resize(ref, tw, th, "disabled")
            ref_latent = video_vae.encode(ref_resized)
            ref_items.append({"type": "image", "data": ref_resized})
            ref_blocks.append(
                {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": ref_latent}
            )
            notes.append(f"Ref2VA reference at {tw}x{th} ({reference_image_size})")

        def exact_chunk_for_shot(si: int, sample_rate: int):
            if exact_audio is None:
                return None
            from mmx_utils.multishot import slice_exact_audio_chunk

            chunk, out_sr, chunk_note = slice_exact_audio_chunk(
                exact_audio["waveform"],
                int(exact_audio["sample_rate"]),
                si,
                frames_per_shot,
                target_rate=sample_rate,
            )
            if chunk_note:
                notes.append(f"shot {si + 1} audio: {chunk_note}")
            return {"waveform": chunk, "sample_rate": out_sr}

        def save_checkpoint(images, waveform, sample_rate, shot_index):
            if not checkpoint_prefix:
                return
            import os
            from fractions import Fraction

            import folder_paths
            from comfy_api.latest import InputImpl, Types

            prefix = f"{checkpoint_prefix}/chain_{shot_index + 1:02d}"
            (full_folder, filename, counter, _subfolder, _prefix) = folder_paths.get_save_image_path(
                prefix, folder_paths.get_output_directory(), width, height
            )
            os.makedirs(full_folder, exist_ok=True)
            path = os.path.join(full_folder, f"{filename}_{counter:05}_.mp4")
            components = Types.VideoComponents(
                images=images.cpu(),
                audio={"waveform": waveform.cpu(), "sample_rate": sample_rate},
                frame_rate=Fraction(24, 1),
            )
            video = InputImpl.VideoFromComponents(components, bit_depth=8)
            video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=18)
            notes.append(f"checkpoint: {path}")

        if start_image is not None:
            prev_last = start_image[:1].clone()
            notes.append("I2V: shot 1 starts from supplied image")

        nested = _nested_factory()
        for si, prompt in enumerate(shots):
            latent, frame_count = mmh3._empty_av_latent(width, height, frames_per_shot)
            exact_chunk = exact_chunk_for_shot(si, 32000)
            if exact_chunk is not None:
                latent = lock_audio_chunk_into_latent(latent, exact_chunk, audio_vae, nested)

            images, keyframes = [], []
            if prev_last is not None:
                crop_mode = "center" if (si == 0 and start_image is not None) else "disabled"
                img = mmh3._resize(prev_last[:1], width, height, crop_mode)
                images.append(img)
                keyframes.append({"resolved_frame_index": 0, "image": img})

            if ref_items and si == 0:
                tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            else:
                tokens = clip.tokenize(prompt, images=images)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks and si == 0:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
            if keyframes:
                for kf in keyframes:
                    kf["latent"] = video_vae.encode(kf.pop("image"))
                cond = node_helpers.conditioning_set_values(
                    cond,
                    {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count},
                )

            import comfy.model_management as _mm

            try:
                clip.patcher.model.to(_mm.text_encoder_offload_device())
            except Exception:
                pass
            try:
                _dev = _mm.get_torch_device()
                _mm.free_memory(_mm.get_total_memory(_dev) * 0.9, _dev)
                _mm.soft_empty_cache()
            except Exception:
                pass

            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            shot_seed = (seed + si) if seed_per_shot else seed
            noise = ncs.RandomNoise().get_noise(shot_seed)[0]
            out, _denoised = ncs.SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)

            lat = out["samples"]
            if getattr(lat, "is_nested", False):
                lat = lat.unbind()[0]
            imgs = video_vae.decode(lat)
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])

            if exact_chunk is not None:
                sr = exact_chunk["sample_rate"]
                wav = exact_chunk["waveform"]
            else:
                aud = vae_decode_audio(audio_vae, out)
                sr = aud["sample_rate"]
                wav = aud["waveform"]

            prev_last = imgs[-1:].clone()
            if si > 0:
                imgs = imgs[1:]
                trim = seam_trim_samples(sr)
                wav = wav[..., trim:].clone()
            save_checkpoint(imgs, wav, sr, si)
            frames_parts.append(imgs.cpu())
            audio_parts.append(wav.cpu())

        master = torch.cat(frames_parts, dim=0)
        if exact_audio is not None:
            waveform = exact_audio["waveform"][:1].clone()
            source_rate = int(exact_audio["sample_rate"])
            if source_rate != sr:
                waveform, _ = resample_waveform(waveform, source_rate, sr)
            target_frames, target_samples, trim_note = trim_master_to_soundtrack(
                master.shape[0], waveform, sr
            )
            master = master[:target_frames]
            waveform = waveform[..., :target_samples]
            notes.append(trim_note)
        else:
            waveform = xfade_audio(audio_parts, sr)

        report = (
            f"{n} shots, {master.shape[0]} frames (~{master.shape[0] / 24.0:.1f}s)"
            + ("; " + "; ".join(notes) if notes else "")
        )
        return io.NodeOutput(
            master,
            {"waveform": waveform, "sample_rate": sr},
            n,
            report,
        )


class MiniMaxH3_MultishotMemorySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_MultishotMemorySampler",
            display_name="H3 Multishot Sampler + Memory (long form)",
            category=CATEGORY,
            description=(
                "Long-form multishot with a memory bank: persistent identity anchor plus recent "
                "shot-end frames for the encoder, while keyframes always continue from the latest "
                "frame for smooth seams."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input(
                    "script",
                    multiline=True,
                    default="Shot 1 prompt goes here.\n---\nShot 2 prompt goes here.",
                ),
                io.Int.Input("shot_count", default=0, min=0, max=64),
                io.Int.Input("width", default=960, min=32, max=4096, step=16),
                io.Int.Input("height", default=544, min=32, max=4096, step=16),
                io.Int.Input("frames_per_shot", default=243, min=5, max=1000),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Int.Input("steps", default=20, min=1, max=50),
                io.Int.Input("memory_frames", default=2, min=0, max=6),
                io.Int.Input("anchor_frames", default=1, min=0, max=2),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Image.Output("master_frames"),
                io.Audio.Output("master_audio"),
                io.Int.Output("shots_rendered"),
                io.String.Output("report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        script,
        shot_count,
        width,
        height,
        frames_per_shot,
        seed,
        steps,
        memory_frames,
        anchor_frames,
        start_image=None,
    ) -> io.NodeOutput:
        import node_helpers
        from comfy_extras import nodes_custom_sampler as ncs
        from comfy_extras import nodes_minimax_h3 as mmh3
        from comfy_extras.nodes_audio import vae_decode_audio

        shots = parse_script(script)
        n = shot_count if shot_count > 0 else len(shots)
        shots, plan_notes = normalize_shot_list(shots, n)

        sigmas = ncs.BasicScheduler().get_sigmas(model, "simple", steps, 1.0)[0]
        sampler = ncs.KSamplerSelect().get_sampler("res_multistep")[0]

        frames_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        sr = None
        history: list[torch.Tensor] = []
        anchor = start_image[:1].clone() if start_image is not None else None
        notes = list(plan_notes)

        import comfy.model_management as _mm

        for si, prompt in enumerate(shots):
            ctx = []
            if anchor is not None and anchor_frames > 0:
                ctx.append(anchor)
            if history:
                take = memory_frames if memory_frames > 0 else 1
                ctx.extend(history[-take:])
            images = [mmh3._resize(c[:1], width, height, "disabled") for c in ctx]

            latent, frame_count = mmh3._empty_av_latent(width, height, frames_per_shot)
            keyframes = []
            cont = history[-1] if history else anchor
            if cont is not None:
                kf = mmh3._resize(cont[:1], width, height, "disabled")
                keyframes.append({"resolved_frame_index": 0, "image": kf})

            tokens = clip.tokenize(prompt, images=images)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if keyframes:
                for kf_ in keyframes:
                    kf_["latent"] = video_vae.encode(kf_.pop("image"))
                cond = node_helpers.conditioning_set_values(
                    cond,
                    {"minimax_keyframes": keyframes, "minimax_frame_count": frame_count},
                )

            try:
                clip.patcher.model.to(_mm.text_encoder_offload_device())
            except Exception:
                pass
            try:
                _dev = _mm.get_torch_device()
                _mm.free_memory(_mm.get_total_memory(_dev) * 0.9, _dev)
                _mm.soft_empty_cache()
            except Exception:
                pass

            guider = ncs.BasicGuider().get_guider(model, cond)[0]
            noise = ncs.RandomNoise().get_noise(seed + si)[0]
            out, _denoised = ncs.SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)

            lat = out["samples"]
            if getattr(lat, "is_nested", False):
                lat = lat.unbind()[0]
            imgs = video_vae.decode(lat)
            if imgs.ndim == 5:
                imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
            aud = vae_decode_audio(audio_vae, out)
            sr = aud["sample_rate"]
            wav = aud["waveform"]

            if anchor is None and anchor_frames > 0:
                anchor = imgs[:1].clone()
                notes.append("identity anchor set from shot 1 frame 1")
            history.append(imgs[-1:].clone())
            if len(history) > 8:
                history.pop(0)

            if si > 0:
                imgs = imgs[1:]
                trim = seam_trim_samples(sr)
                wav = wav[..., trim:].clone()
            frames_parts.append(imgs.cpu())
            audio_parts.append(wav.cpu())

        master = torch.cat(frames_parts, dim=0)
        waveform = torch.cat(audio_parts, dim=-1)
        report = (
            f"{n} shots, {master.shape[0]} frames (~{master.shape[0] / 24.0:.1f}s)"
            + ("; " + "; ".join(notes) if notes else "")
        )
        return io.NodeOutput(master, {"waveform": waveform, "sample_rate": sr}, n, report)
