# PORTED FROM: third_party/ComfyUI-Viggle-Remake/nodes.py (Apache-2.0, Viggle Remake authors); see CREDITS.md and
# THIRD_PARTY_LICENSES. RemakeConditioning / KeyframeUntil / Noise / Prefix reproduce the reference nodes; H3 Remake
# Replace (the anchor -> plate window adapter) is original GPL-3.0 code built on them.
"""H3 Remake: Viggle Remake on ComfyUI's built-in MiniMax H3, plus the Suite's replacement adapter.

Load the Remake LoRA with core LoraLoaderBypassModelOnly (merging it into bf16 weights rounds part of the update away
and the output comes out softer - the reference's own note).

    H3 Remake Conditioning (Viggle)  the source clip as <Video 1> (text-only labels), images as <Picture N>, optional pins
                                     (keyframes), the source soundtrack held clean
    H3 Remake Keyframe Until         drops the keyframes once sigma < keyframe_until
    H3 Remake Noise                  the reference script's noise order, so a seed gives the same video
    H3 Remake Prefix                 the previous take's frames open this render, held clean
    H3 Remake Replace                approved anchor + plate window -> the above, in the window's own framing (zero
                                     offset), chunked by the H3 Job Controller; character, face / head, item / prop

Design: docs/PLAN_h3_replacement_quality.md sections 2-3. Image quality is NOT verified on H3 weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import remake as R  # noqa: E402
from mmx_utils.family_presets import build_six_section_prompt  # noqa: E402

_IMPORT_ERROR: Exception | None = None
try:
    import comfy.audio
    import comfy.nested_tensor
    import comfy.patcher_extension
    import comfy.utils
    import node_helpers
except Exception as _e:  # noqa: BLE001 - the comfy_kitchen skew raises AttributeError, not ImportError
    _IMPORT_ERROR = _e

CATEGORY = "MiniMax H3/Replace"
WRAPPER_KEY = "mmx_remake_keyframe_until"


def _need_comfy(who: str) -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"{who} needs ComfyUI's MiniMax H3 support, which could not be imported ({_IMPORT_ERROR}).")


def _err(who: str, exc: Exception) -> ValueError:
    return ValueError(f"{who}: {exc}")


# ── shared builders (one implementation for the faithful node and the adapter) ─────────────────────────────────────

def _resize(image: torch.Tensor, width: int, height: int, method: str = "lanczos", crop: str = "disabled"):
    return comfy.utils.common_upscale(image.movedim(-1, 1), int(width), int(height), method, crop).movedim(1, -1)


def _cover(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Scale to cover the canvas, then centre-crop with (src - dst) // 2 (the reference's cover)."""
    h, w = image.shape[1], image.shape[2]
    s = max(width / w, height / h)
    sw, sh = max(width, round(w * s)), max(height, round(h * s))
    left, top = (sw - width) // 2, (sh - height) // 2
    return _resize(image, sw, sh)[:, top:top + height, left:left + width]


def _text(clip, nf: int, n_refs: int, prompt: str):
    """Text only: one label per 2 fps block of the source, one per reference, then the prompt, each tokenized alone."""
    segments = R.remake_segments(nf, n_refs, prompt)
    tokens = {"qwen3vl_32b": [[t for s in segments for t in clip.tokenize(s)["qwen3vl_32b"][0]]]}
    return clip.encode_from_tokens_scheduled(tokens)


def _ref_blocks(vae, source: torch.Tensor, refs: list[torch.Tensor], se: int) -> list[dict]:
    """The source on the canvas as the video reference, then the images at their own short edge (rounded to 32)."""
    h, w = int(source.shape[1]), int(source.shape[2])
    z = vae.encode(source)
    blocks = [{"kind": "video", "latent_t": z.shape[2], "latent_h": h // 16, "latent_w": w // 16,
               "ref_audio_t": 0, "latent": z, "audio_latent": None}]
    for r in refs:
        rh, rw = int(r.shape[1]), int(r.shape[2])
        tw, th = R.ref_canvas(rh, rw, se)
        r = r if (rh, rw) == (th, tw) else _resize(r, tw, th)
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(r)})
    return blocks


def _audio_latent(audio_vae, audio, nf: int) -> torch.Tensor:
    wav = R.prepare_wav(audio, nf, comfy.audio.resample)
    return audio_vae.encode(wav[None].movedim(1, -1))[..., :R.audio_latent_len(nf)]


def _condition(positive, blocks: list[dict], keyframes: list[dict]):
    values = {"minimax_refs": blocks, "minimax_keyframes": keyframes} if keyframes else {"minimax_refs": blocks}
    return node_helpers.conditioning_set_values(positive, values)


def _nested(video, za, vmask):
    return {"samples": comfy.nested_tensor.NestedTensor((video, za)),
            "noise_mask": comfy.nested_tensor.NestedTensor((vmask, torch.zeros_like(za)))}


def _apply_prefix(vae, video: torch.Tensor, vmask: torch.Tensor, px: torch.Tensor):
    """Frames px (on the canvas, k = 17m) become this render's first k frames, clean with noise_mask 0. The VAE
    encodes 17-frame clips and drops the last 3 latents: pad to 17m + 5, keep the first 5m."""
    k = int(px.shape[0])
    n = R.prefix_rows(k)
    clean = vae.encode(torch.cat([px, px[-1:].repeat(5, 1, 1, 1)]))[:, :, :n]
    video, vmask = video.clone(), vmask.clone()
    video[:, :, :n] = clean.to(video)
    vmask[:, :, :n] = 0
    return video, vmask


# ── H3 Remake Conditioning (Viggle) ──────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_RemakeConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RemakeConditioning", display_name="H3 Remake Conditioning (Viggle)", category=CATEGORY,
            description="Viggle Remake's conditioning, ported faithfully: the source clip as <Video 1> (text-only "
                        "labels), each image as <Picture N> and, with pins, also a keyframe at that output frame; the "
                        "source soundtrack held clean. Needs the Remake LoRA (Load LoRA bypass). For plate work prefer "
                        "H3 Remake Replace: this node cover-crops off-size sources onto 832x480 / 1280x736.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("source", tooltip="The source clip's frames, at 24 fps. Supplies motion, camera, framing "
                                                 "and everything the prompt does not change."),
                io.String.Input("prompt", multiline=True, tooltip="The six-section prompt, used verbatim."),
                io.Image.Input("ref_1", tooltip="<Picture 1>: usually a frame of the source painted in an image "
                                                "editor."),
                io.String.Input("pins", default="", tooltip="Optional: one output frame per reference image, e.g. "
                                                            "'12, 190'. Each reference is then pinned as a keyframe at "
                                                            "that frame for the first part of the denoise."),
                io.Int.Input("frames", default=124, min=90, max=345, step=17, tooltip="Output length at 24 fps, 17n+5."),
                io.Int.Input("short_edge", default=480, min=480, max=736, step=32,
                             tooltip="480 (832x480 for 16:9) or 736 (1280x736)."),
                io.Audio.Input("audio", optional=True, tooltip="The soundtrack kept in the output, usually the source's "
                                                               "own. Silence when not connected."),
                io.Image.Input("ref_2", optional=True),
                io.Image.Input("ref_3", optional=True),
                io.Image.Input("ref_4", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, source, prompt, ref_1, pins, frames, short_edge, audio=None, ref_2=None,
                ref_3=None, ref_4=None) -> io.NodeOutput:
        who = "H3 Remake Conditioning"
        _need_comfy(who)
        nf, se = int(frames), int(short_edge)
        refs = [r[:1, ..., :3] for r in (ref_1, ref_2, ref_3, ref_4) if r is not None]
        try:
            R.check_frames(nf)
            pin_list = R.parse_pins(pins, len(refs), nf)
        except R.RemakeError as exc:
            raise _err(who, exc) from exc
        source = source[..., :3]
        h, w = int(source.shape[1]), int(source.shape[2])
        canvas = R.viggle_canvas(h, w, se)
        note = "as is"
        if canvas is not None:
            note = f"cover-cropped {w}x{h} -> {canvas[0]}x{canvas[1]}"
            w, h = canvas
            source = _resize(source, w, h, "bicubic", "center")
        source = source[:nf]
        if int(source.shape[0]) != nf:
            raise ValueError(f"{who}: the source has {int(source.shape[0])} frames, {nf} needed.")
        positive = _text(clip, nf, len(refs), prompt)
        blocks = _ref_blocks(vae, source, refs, se)
        keyframes = []
        for i, (r, p) in enumerate(zip(refs, pin_list)):
            if (r.shape[1], r.shape[2]) != (h, w):
                r = _resize(r, w, h) if i == 0 else _cover(r, w, h)      # the first stretched, the rest cover-cropped
            keyframes.append({"resolved_frame_index": p, "latent": vae.encode(r)})
        positive = _condition(positive, blocks, keyframes)
        za = _audio_latent(audio_vae, audio, nf)
        video = torch.zeros(1, 24, R.target_latent_t(nf), h // 16, w // 16)
        report = (f"H3 Remake Conditioning: {nf} frames on {w}x{h} ({note}); {len(refs)} picture(s); "
                  f"pins {pin_list or 'none'}; soundtrack {'held clean' if audio is not None else 'silence'}.")
        return io.NodeOutput(positive, _nested(video, za, torch.ones_like(video)), report)


# ── H3 Remake Keyframe Until ─────────────────────────────────────────────────────────────────────────────────────

def keyframe_until_wrapper(keyframe_until: float):
    def release(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
        p = minimax_payload or {}
        if p.get("keyframes") and float(timestep.flatten()[0]) / 1000 < float(keyframe_until):
            minimax_payload = R.release_payload(p)
        return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    return release


class MiniMaxH3_RemakeKeyframeUntil(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RemakeKeyframeUntil", display_name="H3 Remake Keyframe Until", category=CATEGORY,
            description="Drops Remake's pinned keyframes once sigma falls below keyframe_until (0.9 = the last two of "
                        "six forwards); the references alone remain.",
            inputs=[io.Model.Input("model"),
                    io.Float.Input("keyframe_until", default=0.9, min=0.0, max=1.0, step=0.01,
                                   tooltip="The keyframes condition every forward whose sigma is at least this; below "
                                           "it they are dropped and the references alone remain.")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, keyframe_until=0.9) -> io.NodeOutput:
        _need_comfy("H3 Remake Keyframe Until")
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                               keyframe_until_wrapper(float(keyframe_until)))
        return io.NodeOutput(m)


# ── H3 Remake Noise ──────────────────────────────────────────────────────────────────────────────────────────────

class RemakeNoiseSource:
    """One CPU generator, drawn once per reference and keyframe latent before the target's noise."""

    def __init__(self, seed: int, skip):
        self.seed, self.skip = int(seed), [tuple(s) for s in skip]

    def generate_noise(self, input_latent):
        g = torch.Generator().manual_seed(self.seed)
        for shape in self.skip:
            torch.randn(shape, generator=g)
        samples = input_latent["samples"]
        parts = samples.unbind() if hasattr(samples, "unbind") and not isinstance(samples, torch.Tensor) else [samples]
        noise = [torch.randn(t.shape, generator=g) for t in parts]
        if isinstance(samples, torch.Tensor):
            return noise[0]
        return comfy.nested_tensor.NestedTensor(noise)


class MiniMaxH3_RemakeNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RemakeNoise", display_name="H3 Remake Noise", category=CATEGORY,
            description="Draws the starting noise in Remake's reference order (refs and keyframes first, then the "
                        "target), so a seed gives the same video as the reference script.",
            inputs=[io.Conditioning.Input("positive"),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True)],
            outputs=[io.Noise.Output(display_name="noise")],
        )

    @classmethod
    def execute(cls, positive, seed=0) -> io.NodeOutput:
        _need_comfy("H3 Remake Noise")
        return io.NodeOutput(RemakeNoiseSource(seed, R.noise_skip_shapes(positive[0][1])))


# ── H3 Remake Prefix ─────────────────────────────────────────────────────────────────────────────────────────────

class MiniMaxH3_RemakePrefix(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RemakePrefix", display_name="H3 Remake Prefix", category=CATEGORY,
            description="A hop: the previous take's frames open this render, their latents clean with noise_mask 0 - "
                        "never stepped, they come out unchanged, like the held soundtrack.",
            inputs=[io.Vae.Input("vae"), io.Latent.Input("latent"),
                    io.Image.Input("prefix", tooltip="The previous take's frames, on this render's canvas."),
                    io.Int.Input("prefix_frames", default=17, min=17, max=323, step=17,
                                 tooltip="How many of its frames open this render, a multiple of 17."),
                    io.Int.Input("prefix_at", default=-1, min=-1, max=100000,
                                 tooltip="The previous take's frame they start at; -1: its last prefix_frames.")],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, vae, latent, prefix, prefix_frames=17, prefix_at=-1) -> io.NodeOutput:
        who = "H3 Remake Prefix"
        _need_comfy(who)
        k = int(prefix_frames)
        at = len(prefix) - k if int(prefix_at) < 0 else int(prefix_at)
        px = prefix[at:at + k, ..., :3]
        video, audio = latent["samples"].unbind()
        vmask, amask = latent["noise_mask"].unbind()
        if k % 17 or R.prefix_rows(k) >= video.shape[2]:
            raise ValueError(f"{who}: prefix_frames must be 17m and shorter than the render, got {k}.")
        want = (k, video.shape[3] * 16, video.shape[4] * 16)
        if tuple(px.shape[:3]) != want:
            raise ValueError(f"{who}: the prefix must be {k} frames on the {want[2]}x{want[1]} canvas, got "
                             f"{tuple(px.shape)}.")
        video, vmask = _apply_prefix(vae, video, vmask, px)
        return io.NodeOutput({**latent, "samples": comfy.nested_tensor.NestedTensor((video, audio)),
                              "noise_mask": comfy.nested_tensor.NestedTensor((vmask, amask))})


# ── H3 Remake Replace (the adapter, digest 4.5) ──────────────────────────────────────────────────────────────────

def _to_canvas(images: torch.Tensor, cw: int, ch: int) -> torch.Tensor:
    images = images[..., :3]
    if (int(images.shape[1]), int(images.shape[2])) == (ch, cw):
        return images
    return _resize(images, cw, ch, "bicubic", "disabled")


def _mask_to_canvas(mask: torch.Tensor, nf: int, cw: int, ch: int, who: str) -> torch.Tensor:
    m = mask if mask.ndim == 3 else mask[None]
    if int(m.shape[0]) == 1:
        m = m.expand(nf, -1, -1)
    if int(m.shape[0]) != nf:
        raise ValueError(f"{who}: the mask has {int(m.shape[0])} frames, the window {nf} - one mask per frame.")
    m = m.float()
    if (int(m.shape[1]), int(m.shape[2])) != (ch, cw):
        m = F.interpolate(m[:, None], size=(ch, cw), mode="bilinear", align_corners=False)[:, 0]
    return m.clamp(0, 1)


def _anchor_image(anchor: torch.Tensor, window: torch.Tensor, bboxes, anchor_frame: int, nf: int, who: str):
    a = anchor[:1, ..., :3]
    wh, ww = int(window.shape[1]), int(window.shape[2])
    ah, aw = int(a.shape[1]), int(a.shape[2])
    if (ah, aw) == (wh, ww):
        return a, "anchor at the window's size"
    if bboxes is not None:
        boxes = R.normalize_boxes(bboxes, nf)
        box = boxes[min(max(0, int(anchor_frame)), len(boxes) - 1)]
        return R.crop_box(a, box), f"anchor cut with the window's box of frame {anchor_frame} ({box})"
    if abs(aw / ah - ww / wh) <= 0.01 * (ww / wh):
        return a, f"anchor {aw}x{ah} stretched to the window's framing"
    raise ValueError(f"{who}: the anchor is {aw}x{ah} but the window is {ww}x{wh} with another aspect. When the window "
                     "is a crop (H3 Subject Crop), connect its bboxes so the plate-sized anchor is cut the same way.")


def _context_frames(context: torch.Tensor, window: torch.Tensor, bboxes, nf: int, who: str):
    """The accepted context frames as the window's first k frames: as is at the window's size, or - when the window is
    a crop and the Job Controller hands plate-sized frames - each cut with THIS window's box of the same frame (so a
    crop window that moved since the previous segment still lines up)."""
    ctx = context[..., :3]
    k = int(ctx.shape[0])
    wh, ww = int(window.shape[1]), int(window.shape[2])
    ch_, cw_ = int(ctx.shape[1]), int(ctx.shape[2])
    if (ch_, cw_) == (wh, ww):
        return ctx, "context at the window's size"
    if bboxes is not None:
        boxes = R.normalize_boxes(bboxes, nf)
        if len(boxes) < k:
            raise ValueError(f"{who}: {k} context frames but only {len(boxes)} boxes.")
        cut = [R.crop_box(ctx[t:t + 1], boxes[t]) for t in range(k)]
        return torch.cat([_resize(c, ww, wh, "bicubic", "disabled") for c in cut]), \
            "context cut with this window's boxes"
    if abs(cw_ / ch_ - ww / wh) <= 0.01 * (ww / wh):
        return ctx, "context stretched to the window's framing"
    raise ValueError(f"{who}: the context frames are {cw_}x{ch_} but the window is {ww}x{wh}. When the window is a crop, "
                     "connect H3 Subject Crop's bboxes so the plate-sized context is cut the same way.")


def _pictures(anchor_c: torch.Tensor, references, who: str) -> tuple[list[torch.Tensor], list[str]]:
    pics, notes = [anchor_c], []
    if references is None:
        return pics, notes
    refs = references if references.ndim == 4 else references[None]
    for i in range(int(refs.shape[0])):
        r = refs[i:i + 1, ..., :3]
        if r.shape == anchor_c.shape and torch.equal(r, anchor_c):
            notes.append(f"reference {i} is the anchor itself - used once")
            continue
        if len(pics) >= R.MAX_PICTURES:
            notes.append(f"reference {i} left out: Remake takes at most {R.MAX_PICTURES} pictures")
            continue
        pics.append(r)
    return pics, notes


class MiniMaxH3_RemakeReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_RemakeReplace", display_name="H3 Remake Replace (anchor -> plate window)",
            category=CATEGORY,
            description="Replace a character, a head / face or an item through Viggle Remake: the approved anchor "
                        "(H3 Anchor Approval) and this segment's plate window become Remake conditioning in the "
                        "window's own framing, so <Video 1> sits token-for-token on the output and the result maps "
                        "back onto the plate. First window: the anchor is pinned as a keyframe. Later windows (H3 Job "
                        "Controller context): the accepted tail opens the render, held clean. Mask modes return the "
                        "edit mask and boxes for H3 Plate Restore. Needs the Remake LoRA. Experimental: not A/B-tested "
                        "on H3 weights.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("window", tooltip="This segment's plate frames (H3 Job Controller window -> Frame Window, "
                                                 "or H3 Subject Crop's crops), 17n + 5 frames at 24 fps."),
                io.Image.Input("anchor", tooltip="H3 Anchor Approval's anchor: frame anchor_frame of the take repainted "
                                                 "with the new character / head / item."),
                io.Combo.Input("mode", options=list(R.MODES), default=R.MODES[0],
                               tooltip="character: the performer is replaced everywhere. face / head and item / prop: "
                                       "only inside 'mask' (required)."),
                io.String.Input("prompt", default="", multiline=True,
                                tooltip="Empty: a six-section prompt written for the mode. Otherwise used verbatim."),
                io.String.Input("subject", default="the new character",
                                tooltip="Who / what the anchor shows, for the default prompt."),
                io.Int.Input("short_edge", default=480, min=480, max=736, step=32,
                             tooltip="Remake's trained sizes: 480 or 736. The long side follows the window's aspect."),
                io.Mask.Input("mask", optional=True, tooltip="What is replaced, one mask per window frame (face / head, "
                                                             "item / prop; optional for character)."),
                io.Image.Input("references", optional=True,
                               tooltip="More identity pictures (H3 Reference Bank identity, H3 Subject Memory anchors): "
                                       "EVERY image of the batch is its own <Picture N>, after the anchor; 4 at most."),
                io.Image.Input("context_images", optional=True,
                               tooltip="H3 Job Controller's context_images: the accepted frames this window continues "
                                       "from. 17, 34 ... frames open the render, held clean."),
                io.Audio.Input("audio", optional=True, tooltip="The window's soundtrack: held clean, never re-voiced."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="H3 Subject Crop's boxes when 'window' is a crop: the plate-sized anchor is "
                                             "cut with the box of anchor_frame."),
                io.Int.Input("anchor_frame", default=0, min=0, max=100000, optional=True,
                             tooltip="The window frame the anchor repaints (pinned there on the first window)."),
                io.Boolean.Input("hold_outside_mask", default=False, optional=True,
                                 tooltip="Experimental: hold the plate outside the mask in the latent too (Remake was "
                                         "trained to regenerate the whole target; Plate Restore restores the outside "
                                         "exactly either way)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output(display_name="latent"),
                     io.Image.Output(display_name="plate_canvas", tooltip="The window on the render canvas."),
                     io.Mask.Output(display_name="edit_mask", tooltip="Canvas-space mask for H3 Plate Restore."),
                     io.BoundingBox.Output(display_name="bboxes",
                                           tooltip="Whole-window boxes for H3 Plate Restore (decoded canvas -> window)."),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, window, anchor, mode=R.MODES[0], prompt="", subject="the new character",
                short_edge=480, mask=None, references=None, context_images=None, audio=None, bboxes=None,
                anchor_frame=0, hold_outside_mask=False) -> io.NodeOutput:
        who = "H3 Remake Replace"
        _need_comfy(who)
        window = window if window.ndim == 4 else window[None]
        nf, se = int(window.shape[0]), int(short_edge)
        try:
            R.check_frames(nf)
        except R.RemakeError as exc:
            raise _err(who, exc) from exc
        if mode not in R.MODES:
            raise ValueError(f"{who}: unknown mode {mode!r}.")
        if mode != R.MODES[0] and mask is None:
            raise ValueError(f"{who}: mode '{mode}' replaces only what the mask covers - connect 'mask' (the head / "
                             "item matte, one frame per window frame).")
        wh, ww = int(window.shape[1]), int(window.shape[2])
        cw, ch = R.framing_canvas(wh, ww, se)
        notes = [f"{nf} frames, window {ww}x{wh} -> canvas {cw}x{ch} (stretch "
                 f"{cw / ww:.4f} x {ch / wh:.4f}, no crop: <Video 1> and the output share the canvas)"]
        if not 90 <= nf <= 345:
            notes.append(f"{nf} frames is outside Remake's trained 90-345")
        source = _to_canvas(window, cw, ch)
        anchor_raw, how = _anchor_image(anchor, window, bboxes, int(anchor_frame), nf, who)
        notes.append(how)
        anchor_c = _to_canvas(anchor_raw, cw, ch)
        pics, pic_notes = _pictures(anchor_raw, references, who)
        notes += pic_notes
        # any real context means this is a LATER window: pinning the take's frame-0 anchor here would put frame 0's
        # composition into the middle of the take, so a context that cannot be a prefix is refused, never ignored
        use_prefix = not R.is_placeholder_context(context_images)
        k = int(context_images.shape[0]) if use_prefix else 0
        if use_prefix:
            try:
                R.check_prefix(k, nf)
            except R.RemakeError as exc:
                raise _err(who, exc) from exc
        elif not 0 <= int(anchor_frame) < nf:
            raise ValueError(f"{who}: anchor_frame {anchor_frame} is outside this {nf}-frame window - on the first "
                             "window the anchor is pinned at that frame.")
        text = str(prompt or "").strip() or build_six_section_prompt(R.default_sections(mode, subject,
                                                                                         int(anchor_frame)))
        positive = _text(clip, nf, len(pics), text)
        blocks = _ref_blocks(vae, source, pics, se)
        keyframes = [] if use_prefix else [{"resolved_frame_index": int(anchor_frame), "latent": vae.encode(anchor_c)}]
        positive = _condition(positive, blocks, keyframes)
        za = _audio_latent(audio_vae, audio, nf)
        lt = R.target_latent_t(nf)
        video = torch.zeros(1, 24, lt, ch // 16, cw // 16)
        vmask = torch.ones_like(video)
        edit = _mask_to_canvas(mask, nf, cw, ch, who) if mask is not None else torch.ones(nf, ch, cw)
        if hold_outside_mask and mask is not None:
            from mmx_utils.mask_to_token import pixel_mask_to_h3_latent
            tm = pixel_mask_to_h3_latent(edit, width=cw, height=ch, frame_count=nf)        # [T, h, w]
            video = vae.encode(source).to(video).reshape(video.shape)
            vmask = tm[None, None].to(vmask).expand_as(vmask).clone()
            notes.append(f"plate held outside the mask in the latent (experimental): "
                         f"{float(tm.mean()):.1%} of token rows regenerate")
        if use_prefix:
            ctx, how_ctx = _context_frames(context_images[-k:], window, bboxes, nf, who)
            video, vmask = _apply_prefix(vae, video, vmask, _to_canvas(ctx, cw, ch))
            notes.append(f"window continues a take: its first {k} frames are the accepted context ({how_ctx}), held "
                         "clean; the anchor is <Picture 1>, not pinned")
        else:
            notes.append(f"first window: the anchor is <Picture 1> and pinned at frame {int(anchor_frame)}")
        notes.append(f"{len(pics)} picture(s); mode {mode}; mask "
                     + (f"coverage {float(edit.mean()):.1%}" if mask is not None else "none (whole window)")
                     + f"; soundtrack {'held clean' if audio is not None else 'silence'}")
        report = json.dumps({"node": who, "mode": mode, "frames": nf, "canvas": [cw, ch], "window": [ww, wh],
                             "pinned": [] if use_prefix else [int(anchor_frame)], "prefix_frames": k,
                             "pictures": len(pics), "notes": notes}, indent=1)
        return io.NodeOutput(positive, _nested(video, za, vmask), source.clamp(0, 1), edit,
                             R.full_boxes(nf, ww, wh), report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3_RemakeConditioning": MiniMaxH3_RemakeConditioning,
    "MiniMaxH3_RemakeKeyframeUntil": MiniMaxH3_RemakeKeyframeUntil,
    "MiniMaxH3_RemakeNoise": MiniMaxH3_RemakeNoise,
    "MiniMaxH3_RemakePrefix": MiniMaxH3_RemakePrefix,
    "MiniMaxH3_RemakeReplace": MiniMaxH3_RemakeReplace,
}
