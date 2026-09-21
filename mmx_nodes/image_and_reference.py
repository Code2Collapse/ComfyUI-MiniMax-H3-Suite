"""H3 Image + Reference to Video — keyframes and references in one node.

Ported from ComfyUi-MiniMax-H3-Image-And-Reference-To-Video.

H3 takes two kinds of image conditioning and the stock nodes make you choose:
MiniMaxH3ImageToVideo does keyframes, MiniMaxH3ReferenceToVideo does
references. A shot usually wants both - "start here, end there, and it is THIS
person throughout" - so this builds one conditioning payload carrying both.

The private helpers from comfy_extras.nodes_minimax_h3 are deliberately reused
rather than reimplemented: they are what the built-in nodes call, and a second
copy of the canvas arithmetic would drift from them at the next ComfyUI
update. They are private names, so the import is guarded and the failure is
explained rather than crashing the pack's import.
"""

from __future__ import annotations

import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.reference_prep import (
    CANVAS_MULTIPLE,
    ReferencePrepError,
    canvas_size,
    check_reference_order,
    describe,
    reference_scale,
    sample_indices,
    sample_timestamps,
    snap_to_latent_grid,
    soundtrack_key,
)

_IMPORT_ERROR: Exception | None = None
try:
    import node_helpers
    import nodes as _comfy_nodes
    from comfy_extras.nodes_minimax_h3 import (
        _empty_av_latent,
        _encode_ref_audio,
        _resize,
        adapt_canvas,
    )
    from comfy_extras.nodes_minimax_h3 import CANVAS_MULTIPLE as _CORE_MULTIPLE
except Exception as _e:  # noqa: BLE001 - private names; see the docstring
    _IMPORT_ERROR = _e
    node_helpers = None  # type: ignore[assignment]
    _comfy_nodes = None  # type: ignore[assignment]
    _empty_av_latent = _encode_ref_audio = _resize = adapt_canvas = None  # type: ignore[assignment]
    _CORE_MULTIPLE = CANVAS_MULTIPLE

MAX_RESOLUTION = getattr(_comfy_nodes, "MAX_RESOLUTION", 16384)


def _names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(1, count + 1)]


class MiniMaxH3_ImageAndReferenceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ImageAndReferenceToVideo",
            display_name="H3 Image + Reference to Video",
            category="MiniMax H3/Conditioning",
            description=(
                "Keyframes AND references in one conditioning. Keyframes "
                "(first/last frame) occupy real frame positions and say where "
                "the shot starts and ends; references (<Picture n>, <Video n>, "
                "<Audio n>) have no position and say who and what sort. The "
                "stock nodes make you pick one; most shots want both."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frames at 24 fps, snapped up to H3's 17k+5 grid. "
                            "124 is about 5 seconds; the model was trained "
                            "around 124-362 and longer is untested, not "
                            "forbidden."),
                io.Image.Input(
                    "first_frame", optional=True,
                    tooltip="Frame 0 of the output. Stretched to the canvas, "
                            "because it is the geometry anchor - cropping it "
                            "would move the shot."),
                io.Image.Input(
                    "last_frame", optional=True,
                    tooltip="The final frame. Cover-cropped to preserve its "
                            "aspect, because it follows the anchor rather than "
                            "setting the geometry."),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    optional=True,
                    tooltip="'match' scales each reference to the generation's "
                            "pixel area. 'max' uses a 2048px short edge for the "
                            "best identity fidelity and is SEVERAL TIMES "
                            "SLOWER - reference tokens ride through every "
                            "sampling step, not just the first."),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Identity/style reference. Downscaled if "
                                    "large, never upscaled."),
                        names=_names("ref_image", 10), min=0)),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Motion/style reference at 24 fps, 2-15s."),
                        names=_names("ref_video", 4), min=0)),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the reference video with the "
                                    "SAME NUMBER. Paired by number, not by "
                                    "position - leaving video 2 empty does not "
                                    "shift video 3's audio onto video 1."),
                        names=_names("ref_video_audio", 4), min=0)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_audio", tooltip="Standalone voice reference."),
                        names=_names("ref_audio", 4), min=0)),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.String.Output(
                    display_name="report",
                    tooltip="What was built: how many of each reference, what "
                            "the length snapped to, and where keyframes and "
                            "reference videos will fight."),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                first_frame=None, last_frame=None, ref_image_size="match",
                ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None) -> io.NodeOutput:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "H3 Image + Reference to Video needs ComfyUI's built-in MiniMax "
                f"H3 nodes, which could not be imported ({_IMPORT_ERROR}). This "
                "ComfyUI build is either too old for H3 or has the H3 extras "
                "disabled.")
        if _CORE_MULTIPLE != CANVAS_MULTIPLE:
            raise RuntimeError(
                f"ComfyUI's H3 canvas granularity changed ({_CORE_MULTIPLE} vs "
                f"the {CANVAS_MULTIPLE} this node was written against). Every "
                "reference would be sized to the wrong grid, so this stops "
                "rather than producing quietly misaligned conditioning.")

        latent, frame_count = _empty_av_latent(width, height, length)

        # ── keyframes (fl2va) ───────────────────────────────────────────────
        images = []
        keyframes = []
        if first_frame is not None:
            # Stretched, not cropped: this frame defines the framing, and a
            # crop here moves the whole shot.
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # Cover-cropped: it follows the anchor, so its aspect is preserved
            # rather than the framing.
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        # ── references (ref2va) ─────────────────────────────────────────────
        # Both lists are appended to in lockstep. The encoder numbers
        # references by the order of `ref_items`; the DiT indexes `ref_blocks`.
        # They must not drift - see check_reference_order.
        ref_items: list[dict] = []
        ref_blocks: list[dict] = []
        n_images = n_videos = n_audios = 0

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = int(img.shape[1]), int(img.shape[2])
            scale = reference_scale(width, height, w, h, ref_image_size)
            tw, th = canvas_size(w, h, scale)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image",
                "latent_h": th // CANVAS_MULTIPLE,
                "latent_w": tw // CANVAS_MULTIPLE,
                "latent": vae.encode(resized),
            })
            n_images += 1

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get(soundtrack_key(name))
            vh, vw = int(video_frames.shape[1]), int(video_frames.shape[2])
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw, ch = canvas_size(vw, vh, 1.0)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            frames = frames[:snap_to_latent_grid(int(frames.shape[0]))]

            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
                n_audios += 1

            idx = sample_indices(int(frames.shape[0]))
            ref_items.append({
                "type": "video",
                "data": frames[idx],
                "timestamps": sample_timestamps(idx),
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2],
                "latent_h": ch // CANVAS_MULTIPLE,
                "latent_w": cw // CANVAS_MULTIPLE,
                "ref_audio_t": ref_audio_t,
                "latent": z,
                "audio_latent": audio_latent,
            })
            n_videos += 1

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})
            n_audios += 1

        check_reference_order(ref_items, ref_blocks)

        # Keyframe images go into the vision context AFTER the references, so
        # reference numbering stays stable: a prompt saying "<Picture 2>" must
        # keep meaning the second ref_image whether or not a keyframe is wired.
        encoder_items = ref_items + [{"type": "image", "data": im} for im in images]
        tokens = clip.tokenize(prompt, minimax_ref_items=encoder_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        payload: dict = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            payload["minimax_keyframes"] = keyframes
        if ref_blocks:
            payload["minimax_refs"] = ref_blocks
        if payload:
            cond = node_helpers.conditioning_set_values(cond, payload)

        report = describe(len(keyframes), n_images, n_videos, n_audios,
                          ref_image_size, int(length), int(frame_count))
        return io.NodeOutput(cond, latent, report)


__all__ = ["MiniMaxH3_ImageAndReferenceToVideo", "ReferencePrepError"]
