# PORTED FROM: https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop@43afeff chain_nodes.py
# (MiniMaxH3TaggedPictureReference, MiniMaxH3TaggedVideoReference, MiniMaxH3TaggedAudioReference,
# MiniMaxH3TaggedReferenceToVideo), GPL-3.0, Copyright (C) 2026 NikoDemon80 / ethanfel. Backend only: upstream's Scene
# Prompt Editor @ tray, carousel and Plan studio front end are not ported. Changes: see mmx_utils/tagged_refs.py.
"""H3 Tagged Picture / Video / Audio + H3 Tagged Reference to Video: write @tags in the prompt; only the references
it mentions reach H3, under core Reference to Video's own <Picture N> / <Video N> / <Audio N> labels."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import tagged_refs as TR  # noqa: E402

CATEGORY = "MiniMax H3/Conditioning"
TAGGED_REFS = io.Custom("MMX_H3_TAGGED_REFERENCES")
CHAIN_TIP = "Optional earlier Tagged node: chain them in the order you want them numbered."


def _image(x, label, min_frames=1):
    if not torch.is_tensor(x) or x.ndim != 4 or int(x.shape[0]) < min_frames or int(x.shape[-1]) < 3:
        raise TR.TaggedRefError(f"{label} must be an IMAGE [frames, height, width, channels] with at least "
                                f"{min_frames} frame(s).")


def _out(refs, what):
    status = f"registered {what}; {TR.describe(refs)}"
    return io.NodeOutput(refs, status, ui={"text": [status]})


class MiniMaxH3_TaggedPicture(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TaggedPicture", display_name="H3 Tagged Picture", category=CATEGORY,
            description=("Register one reference picture under an @tag. Write @tag in the H3 Tagged Reference to "
                         "Video prompt to use it; a picture no prompt mentions is never sent to H3."),
            search_aliases=["tagged reference", "@tag picture", "h3 reference picture"],
            inputs=[io.Image.Input("image", tooltip="The picture. Only the first image of a batch is used."),
                    io.String.Input("tag", default="hero_face", tooltip="Write @hero_face in the prompt to use it."),
                    TAGGED_REFS.Input("previous", optional=True, tooltip=CHAIN_TIP)],
            outputs=[TAGGED_REFS.Output(display_name="references"), io.String.Output(display_name="status")])

    @classmethod
    def execute(cls, image, tag, previous=None) -> io.NodeOutput:
        _image(image, "H3 Tagged Picture")
        pic = image[:1]
        refs = TR.append(previous, "picture", tag, pic, TR.tensor_fingerprint(pic))
        return _out(refs, f"@{refs['entries'][-1]['tag']} picture")


class MiniMaxH3_TaggedVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TaggedVideo", display_name="H3 Tagged Video", category=CATEGORY,
            description=("Register one reference video (24 fps frames) and, optionally, its soundtrack under @tags. "
                         "Mentioning either tag in the prompt sends the pair; the soundtrack gets its own <Audio N> "
                         "label right before the video's <Video N>."),
            search_aliases=["tagged reference video", "@tag video", "h3 reference video"],
            inputs=[io.Image.Input("video", tooltip="Reference frames at 24 fps (at least 5)."),
                    io.String.Input("tag", default="performance", tooltip="Write @performance in the prompt."),
                    io.String.Input("audio_tag", default="",
                                    tooltip="Tag of the soundtrack. Blank: @<tag>_audio."),
                    io.Audio.Input("audio", optional=True, tooltip="The video's own soundtrack."),
                    TAGGED_REFS.Input("previous", optional=True, tooltip=CHAIN_TIP)],
            outputs=[TAGGED_REFS.Output(display_name="references"), io.String.Output(display_name="status")])

    @classmethod
    def execute(cls, video, tag, audio_tag="", audio=None, previous=None) -> io.NodeOutput:
        _image(video, "H3 Tagged Video", 5)
        a_hash = ""
        if audio is not None:
            TR.validate_audio(audio, "H3 Tagged Video soundtrack")
            a_hash = TR.audio_fingerprint(audio)
        refs = TR.append(previous, "video", tag, video, TR.tensor_fingerprint(video), audio=audio,
                         audio_tag=audio_tag, audio_hash=a_hash)
        e = refs["entries"][-1]
        return _out(refs, f"@{e['tag']} video" + (f" + @{e['audio_tag']} soundtrack" if e.get("audio_tag") else ""))


class MiniMaxH3_TaggedAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TaggedAudio", display_name="H3 Tagged Audio", category=CATEGORY,
            description="Register a standalone voice / music / sound reference under an @tag.",
            search_aliases=["tagged reference audio", "@tag audio", "h3 reference audio"],
            inputs=[io.Audio.Input("audio", tooltip="Standalone reference audio."),
                    io.String.Input("tag", default="voice", tooltip="Write @voice in the prompt."),
                    TAGGED_REFS.Input("previous", optional=True, tooltip=CHAIN_TIP)],
            outputs=[TAGGED_REFS.Output(display_name="references"), io.String.Output(display_name="status")])

    @classmethod
    def execute(cls, audio, tag, previous=None) -> io.NodeOutput:
        TR.validate_audio(audio, "H3 Tagged Audio")
        refs = TR.append(previous, "audio", tag, audio, TR.audio_fingerprint(audio))
        return _out(refs, f"@{refs['entries'][-1]['tag']} audio")


def _core_ref2va():
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    return MiniMaxH3ReferenceToVideo


class MiniMaxH3_TaggedReferenceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_TaggedReferenceToVideo", display_name="H3 Tagged Reference to Video",
            category=CATEGORY,
            description=("Core MiniMax H3 Reference to Video, driven by @tags. Only the references the prompt "
                         "mentions are sent, numbered as core numbers them (pictures, then each video with its "
                         "soundtrack's <Audio N> first, then standalone audio), and every @tag is replaced by its "
                         "native label. H3 takes at most 9 pictures, 3 videos and 3 standalone audios."),
            search_aliases=["tagged ref2va", "@tag reference to video", "prompt references"],
            inputs=[
                io.Clip.Input("clip"),
                TAGGED_REFS.Input("references", tooltip="The chain of H3 Tagged Picture / Video / Audio nodes."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Write @tags where the reference belongs, e.g. '<Subject 1> has the face of "
                                        "@hero_face and moves like @performance.'"),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32),
                io.Int.Input("height", default=768, min=32, max=8192, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps (17k + 5)."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="As core: match scales pictures to the generation's area, max keeps up to "
                                       "2048 px on the short edge (slower)."),
                io.Combo.Input("compliance", options=list(TR.COMPLIANCE), default="strict",
                               tooltip="strict: an unknown @tag or too many references stop the run. soft: they "
                                       "become warnings, unknown @tags stay in the prompt, the first references in "
                                       "chain order are kept. disabled: @tags are left as written."),
                io.Vae.Input("vae", optional=True, tooltip="H3 video VAE (without it references only reach the "
                                                           "text encoder)."),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent"),
                     io.String.Output(display_name="compiled_prompt"), io.String.Output(display_name="report")])

    @classmethod
    def execute(cls, clip, references, prompt, width=1344, height=768, length=124, ref_image_size="match",
                compliance="strict", vae=None, audio_vae=None) -> io.NodeOutput:
        compiled, b, report = TR.compile_prompt(references, prompt, compliance)
        out = _core_ref2va().execute(clip=clip, prompt=compiled, width=width, height=height, length=length,
                                     ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                                     **TR.ref2va_inputs(b))
        res = getattr(out, "result", None) or getattr(out, "args", None) or out
        positive, latent = res[0], res[1]
        return io.NodeOutput(positive, latent, compiled, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_TaggedPicture", "MiniMaxH3_TaggedVideo", "MiniMaxH3_TaggedAudio",
           "MiniMaxH3_TaggedReferenceToVideo"]
