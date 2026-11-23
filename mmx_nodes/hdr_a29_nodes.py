"""MiniMax H3 HDR - inference and I/O nodes (owner order A29, 2026-11-16; design docs/hdr/A29_DESIGN.md section 4 in the
C2C workspace). Working space: scene-linear Rec.709, 1.0 = diffuse white. Encoding: diffhdr_log_16384.

Every formula comes from mmx_utils/hdr_math.py (the single source of truth); files go through mmx_utils/hdr_a29_io.py.
Linear light is never clamped to 1 after Log Decode: only the log signal (Log Encode / Save Log 16-bit) is held to
0..1, half-float EXR saturates at 65504 (the format's maximum) and HDR10 at 10000 nits (PQ's maximum) - both counted
in the report.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from comfy_api.latest import io, ui

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import hdr_a29_io as HIO  # noqa: E402
from mmx_utils import hdr_math as HM  # noqa: E402

LOG = logging.getLogger("MiniMaxSuite.hdr_a29")
CATEGORY = "MiniMax H3/HDR"
GAMUT_NAMES = list(HM.GAMUTS)
NO_ENCODE = "none (already log)"
DEFAULT_TEMPLATE = ("{scene}. Lighting: {lighting}. Camera: {camera}. High dynamic range footage, scene-linear light, "
                    "detail kept in bright highlights and deep shadows.")
PREVIEW_MAX_FRAMES = 16


# ---------------------------------------------------------------------------------------------------------- helpers
def _img(x, what="image") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{what}: connect an IMAGE")
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[-1] < 3 or x.shape[0] == 0:
        raise ValueError(f"{what}: expected IMAGE [frames, H, W, 3], got shape {tuple(x.shape)}")
    return x[..., :3].float()


def _stats(x: torch.Tensor) -> str:
    return f"min {float(x.min()):.4g}, max {float(x.max()):.4g}x white, {float((x > 1).float().mean()) * 100:.2f} % above 1"


def _out_root() -> str:
    import folder_paths
    return os.path.abspath(folder_paths.get_output_directory())


def _split_prefix(prefix: str) -> tuple[str, str]:
    """(subfolder, name) of a save prefix, refused when it leaves the output folder."""
    raw = str(prefix or "").replace("\\", "/").strip()
    p = raw.strip("/")
    if not p:
        raise ValueError("filename_prefix is empty: give a name such as 'hdr/exr/shot010'")
    if raw.startswith("/") or os.path.isabs(p) or re.match(r"^[A-Za-z]:", p) or any(part == ".." for part in p.split("/")):
        raise ValueError(f"filename_prefix '{prefix}' must stay inside ComfyUI's output folder (no absolute path, no '..')")
    sub, _, name = p.rpartition("/")
    if not re.match(r"^[\w.\- ]+$", name):
        raise ValueError(f"filename_prefix name '{name}' may hold letters, digits, space, '.', '-' and '_' only")
    return sub, name


def _new_target(prefix: str, ext: str | None = None) -> tuple[str, str]:
    """(path, name): a NEW folder <output>/<sub>/<name>_NNNNN/ (sequences) or file <output>/<sub>/<name>_NNNNN<ext>."""
    sub, name = _split_prefix(prefix)
    root = _out_root()
    base = os.path.abspath(os.path.join(root, sub))
    if os.path.commonpath([root, base]) != root:
        raise ValueError(f"filename_prefix '{prefix}' leaves the output folder")
    os.makedirs(base, exist_ok=True)
    for n in range(1, 100000):
        cand = os.path.join(base, f"{name}_{n:05d}" + (ext or ""))
        if not os.path.exists(cand):
            if ext is None:
                os.makedirs(cand)
            return cand, name
    raise RuntimeError(f"no free counter left for {prefix}")


def _rel(path: str) -> str:
    try:
        return "output/" + os.path.relpath(path, _out_root()).replace("\\", "/")
    except ValueError:
        return path


def _natural(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# ---------------------------------------------------------------------------------------------------- 1 encode/decode
class MiniMaxH3_HDRLogEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRLogEncode",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Log Encode",
            category=CATEGORY,
            description=("Scene-linear HDR (1.0 = diffuse white) -> log code values 0..1 that H3's VAE can carry. "
                         "Default diffhdr_log_16384 (DiffHDR log-gamma 2.2, holds up to 16384x white)."),
            search_aliases=["hdr log encode", "linear to log", "diffhdr encode"],
            inputs=[
                io.Image.Input("image", tooltip="Scene-linear Rec.709 frames; values above 1 are kept up to the curve's maximum."),
                io.Combo.Input("encoding", options=HM.SUITE_ENCODINGS, default=HM.DEFAULT_ENCODING,
                               tooltip="Log curve. diffhdr_log_16384 is the A29 standard (training and inference must match)."),
            ],
            outputs=[io.Image.Output("image", tooltip="Log code values 0..1 (IMAGE, one frame per input frame); linear "
                                                      "values above the curve maximum are held at 1. Feed to VAE Encode "
                                                      "or Save Log 16-bit.")],
        )

    @classmethod
    def execute(cls, image, encoding=HM.DEFAULT_ENCODING) -> io.NodeOutput:
        lin = _img(image)
        code = HM.encode(lin, encoding)
        over = float((lin > HM.enc_max(encoding)).any(-1).float().mean()) * 100
        text = f"{encoding}: {lin.shape[0]} frames; input {_stats(lin)}; {over:.3f} % of pixels above the curve maximum " \
               f"({HM.enc_max(encoding):.4g}x) were held at it"
        return io.NodeOutput(code.contiguous(), ui={"text": [text]})


class MiniMaxH3_HDRLogDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRLogDecode",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Log Decode",
            category=CATEGORY,
            description=("Log code values (the H3 VAE decode of a log-HDR generation) -> scene-linear HDR, 1.0 = diffuse "
                         "white. The output is NOT clamped: values above 1 are the HDR highlights."),
            search_aliases=["hdr log decode", "log to linear", "diffhdr decode"],
            inputs=[
                io.Image.Input("image", tooltip="Log code values 0..1 (VAE Decode output of the HDR LoRA path)."),
                io.Combo.Input("encoding", options=HM.SUITE_ENCODINGS, default=HM.DEFAULT_ENCODING,
                               tooltip="The curve the LoRA was trained with (A29: diffhdr_log_16384)."),
            ],
            outputs=[io.Image.Output("image", tooltip="Scene-linear Rec.709 HDR (IMAGE, 1.0 = diffuse white), NOT "
                                                      "clamped: values above 1 are highlights. Next: Bias Correct, "
                                                      "Gain Map Join or a Save node.")],
        )

    @classmethod
    def execute(cls, image, encoding=HM.DEFAULT_ENCODING) -> io.NodeOutput:
        lin = HM.decode(_img(image), encoding).contiguous()
        return io.NodeOutput(lin, ui={"text": [f"{encoding} -> linear: {_stats(lin)} (not clamped)"]})


# ---------------------------------------------------------------------------------------------------- 2 bias correct
class MiniMaxH3_HDRBiasCorrect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRBiasCorrect",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Bias Correct",
            category=CATEGORY,
            description=("Undoes the H3 VAE's known exposure bias: every pixel is multiplied by 2^(-bias(luma) x strength), "
                         "the bias looked up in an editable 'luma: stops' table (log-luma interpolation). Luminance only: "
                         "colour ratios are unchanged. Default table = the owner's sheet, diffhdr_log_16384 column."),
            search_aliases=["hdr bias correct", "vae bias", "exposure bias"],
            inputs=[
                io.Image.Input("image", tooltip="Scene-linear HDR after Log Decode."),
                io.String.Input("table", default=HM.DEFAULT_BIAS_TABLE, multiline=True,
                                tooltip="One 'luma: stops' line per point (luma linear, 1.0 = diffuse white; stops the "
                                        "VAE ADDS, negative = it darkens). '#' starts a comment. Flat outside the table."),
                io.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.01,
                               tooltip="1 = apply the table as written, 0 = off."),
            ],
            outputs=[io.Image.Output("image", tooltip="Scene-linear HDR, each pixel scaled by 2^(-bias(luma) x "
                                                      "strength); colour ratios unchanged, not clamped."),
                     io.String.Output("report", tooltip="Text: parsed table points, strength, mean correction in "
                                                        "stops and min/max/% above 1 of the output.")],
        )

    @classmethod
    def execute(cls, image, table=HM.DEFAULT_BIAS_TABLE, strength=1.0) -> io.NodeOutput:
        lin = _img(image)
        pts = HM.parse_bias_table(table)
        out = HM.bias_correct(lin, pts, strength).contiguous()
        gain = HM.luma709(out).clamp_min(1e-8) / HM.luma709(lin).clamp_min(1e-8)
        valid = HM.luma709(lin) > 1e-6
        mean_stops = float(torch.log2(gain[valid]).mean()) if bool(valid.any()) else 0.0
        report = (f"bias table: {len(pts)} points ({', '.join(f'{a:g}: {b:+g}' for a, b in pts)}), strength {strength:g}; "
                  f"mean correction {mean_stops:+.3f} stops; output {_stats(out)}")
        return io.NodeOutput(out, report, ui={"text": [report]})


# ----------------------------------------------------------------------------------------------------- 3 gamut
class MiniMaxH3_HDRGamutConvert(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRGamutConvert",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Gamut Convert",
            category=CATEGORY,
            description="Linear RGB 3x3 matrix between Rec.709, Rec.2020 and ACEScg (AP1). No clamp: out-of-gamut values stay.",
            search_aliases=["hdr gamut", "rec2020", "acescg", "primaries"],
            inputs=[
                io.Image.Input("image", tooltip="Scene-linear frames (not log codes)."),
                io.Combo.Input("source", options=GAMUT_NAMES, default="rec709", tooltip="Primaries of the input."),
                io.Combo.Input("target", options=GAMUT_NAMES, default="rec2020", tooltip="Primaries of the output."),
            ],
            outputs=[io.Image.Output("image", tooltip="Linear RGB in the target primaries (3x3 matrix applied); not "
                                                      "clamped, so out-of-gamut negatives and values above 1 stay.")],
        )

    @classmethod
    def execute(cls, image, source="rec709", target="rec2020") -> io.NodeOutput:
        out = HM.gamut_convert(_img(image), source, target).contiguous()
        return io.NodeOutput(out, ui={"text": [f"{HM.GAMUTS[source]} -> {HM.GAMUTS[target]}"]})


# ----------------------------------------------------------------------------------------------------- 4 clip mask
def _grow_feather(m: torch.Tensor, grow: int, feather: int) -> torch.Tensor:
    x = m[:, None]
    if grow > 0:
        x = torch.nn.functional.max_pool2d(x, 2 * grow + 1, 1, grow)
    if feather > 0:
        x = torch.nn.functional.avg_pool2d(x, 2 * feather + 1, 1, feather, count_include_pad=False)
    return x[:, 0].clamp(0.0, 1.0)


class MiniMaxH3_HDRClipMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRClipMask",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Clip Mask",
            category=CATEGORY,
            description=("Where the SDR source is clipped: over = any channel at or above 'over'; under = luma at or below "
                         "'under'. Optional grow (dilate) and feather (soft edge) in pixels."),
            search_aliases=["hdr clip mask", "overexposed mask", "blown highlights", "crushed shadows"],
            inputs=[
                io.Image.Input("image", tooltip="The SDR picture (0..1 sRGB code values)."),
                io.Float.Input("over", default=0.98, min=0.0, max=1.0, step=0.001,
                               tooltip="Over-exposed when any channel is at or above this code value."),
                io.Float.Input("under", default=0.02, min=0.0, max=1.0, step=0.001,
                               tooltip="Under-exposed when the luma of the code values is at or below this."),
                io.Int.Input("grow", default=0, min=0, max=256, tooltip="Dilate both masks by this many pixels."),
                io.Int.Input("feather", default=0, min=0, max=256, tooltip="Soften the edges over this many pixels."),
            ],
            outputs=[io.Mask.Output("over_mask", tooltip="MASK, 1 = over-exposed (any channel >= 'over'), after grow "
                                                         "and feather; one mask per frame."),
                     io.Mask.Output("under_mask", tooltip="MASK, 1 = under-exposed (Rec.709 luma of the code values "
                                                          "<= 'under'), after grow and feather; one mask per frame.")],
        )

    @classmethod
    def execute(cls, image, over=0.98, under=0.02, grow=0, feather=0) -> io.NodeOutput:
        o, u = HM.clip_masks(_img(image), over, under)
        o, u = _grow_feather(o, int(grow), int(feather)), _grow_feather(u, int(grow), int(feather))
        text = f"over {float(o.mean()) * 100:.2f} % / under {float(u.mean()) * 100:.2f} % of the area"
        return io.NodeOutput(o, u, ui={"text": [text]})


# -------------------------------------------------------------------------------------------------- 5 gain-map join
def gain_map_join(draft_hdr, regen, eps=1e-3, radius=4, guide_eps=1e-3):
    d, r = _img(draft_hdr, "draft_hdr"), _img(regen, "regen")
    if d.shape[0] != r.shape[0]:
        raise ValueError(f"frame counts differ: draft HDR {d.shape[0]}, Regenerate output {r.shape[0]} - trim them to "
                         "the same frames first (Regenerate Finish frame_count)")
    if r.shape[1] < d.shape[1] or r.shape[2] < d.shape[2]:
        raise ValueError(f"the Regenerate output ({r.shape[2]}x{r.shape[1]}) is smaller than the draft HDR "
                         f"({d.shape[2]}x{d.shape[1]}): connect the 4K output to 'regen'")
    frames, first = [], ""
    with torch.no_grad():
        for i in range(d.shape[0]):
            o, rep = HM.gain_map_join(d[i:i + 1], r[i:i + 1], float(eps), int(radius), float(guide_eps))
            frames.append(o)
            first = first or rep
    out = torch.cat(frames).contiguous()
    return out, f"{first}\n{out.shape[0]} frames; output {_stats(out)}"


class MiniMaxH3_HDRGainMapJoin(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRGainMapJoin",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Gain Map Join",
            category=CATEGORY,
            description=("Draft-resolution linear HDR + the Regenerate 4K SDR output -> 4K linear HDR: gain = hdr / "
                         "max(downscale(regen_linear), eps), upsampled to 4K with a guided filter on the 4K luma, times "
                         "the 4K linear picture. Detail from Regenerate, light from the HDR draft. Not clamped."),
            search_aliases=["hdr gain map", "hdr 4k join", "regenerate hdr"],
            inputs=[
                io.Image.Input("draft_hdr", tooltip="Linear HDR at draft size (Log Decode / Bias Correct output)."),
                io.Image.Input("regen", tooltip="Regenerate STD 4K output of the SAME SDR clip (sRGB 0..1), same frames."),
                io.Float.Input("eps", default=1e-3, min=1e-6, max=1.0, step=1e-4,
                               tooltip="Floor of the linear Regenerate picture in the gain division (protects black)."),
                io.Int.Input("radius", default=4, min=1, max=64,
                             tooltip="Guided-filter window radius at draft resolution: larger = smoother gain."),
                io.Float.Input("guide_eps", default=1e-3, min=1e-6, max=1.0, step=1e-4,
                               tooltip="Guided-filter regularisation: larger = less edge-following."),
            ],
            outputs=[io.Image.Output("image", tooltip="Linear HDR at the 'regen' resolution (sRGB-decoded regen x "
                                                      "upsampled gain), one frame per input frame; not clamped. Save "
                                                      "with Save EXR Sequence or Save HDR10."),
                     io.String.Output("report", tooltip="Text: draft and regen sizes, gain range, frame count and "
                                                        "output min/max/% above 1.")],
        )

    @classmethod
    def execute(cls, draft_hdr, regen, eps=1e-3, radius=4, guide_eps=1e-3) -> io.NodeOutput:
        out, report = gain_map_join(draft_hdr, regen, eps, radius, guide_eps)
        return io.NodeOutput(out, report, ui={"text": [report]})


# ------------------------------------------------------------------------------------------------------- 6 prompt
def build_prompt(scene: str, lighting: str, camera: str, template: str = DEFAULT_TEMPLATE) -> str:
    fields = {"scene": str(scene or "").strip(), "lighting": str(lighting or "").strip(),
              "camera": str(camera or "").strip()}
    text = str(template or DEFAULT_TEMPLATE)
    # a sentence whose only content is an empty field ("Lighting: {lighting}.") is dropped
    for k, v in fields.items():
        if not v:
            text = re.sub(r"[^.\n{}]*\{" + k + r"\}[ \t]*[.;,]?", "", text)
    for k, v in fields.items():
        text = text.replace("{" + k + "}", v)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\. ?\.", ".", text)
    return "\n".join(line.strip().lstrip(". ").strip() for line in text.splitlines()).strip()


class MiniMaxH3_HDRPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRPrompt",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Prompt",
            category=CATEGORY,
            description=("Builds the HDR caption / inference prompt from three fields and a template ({scene}, "
                         "{lighting}, {camera}). Empty fields drop their sentence. Use the same template for the "
                         "training captions and for inference."),
            search_aliases=["hdr prompt", "hdr caption"],
            inputs=[
                io.String.Input("scene", default="", multiline=True, tooltip="What is in the shot."),
                io.String.Input("lighting", default="", multiline=True, tooltip="Light sources and contrast."),
                io.String.Input("camera", default="", multiline=True, tooltip="Lens, framing, movement."),
                io.String.Input("template", default=DEFAULT_TEMPLATE, multiline=True,
                                tooltip="Placeholders {scene} {lighting} {camera}. The owner's order (A29) named a "
                                        "template that was missing from the message, so this default is a stand-in: "
                                        "edit it to your wording."),
            ],
            outputs=[io.String.Output("prompt", tooltip="The filled template text (sentences whose field is empty are "
                                                        "removed); wire to the text encoder or use as a training "
                                                        "caption.")],
        )

    @classmethod
    def execute(cls, scene="", lighting="", camera="", template=DEFAULT_TEMPLATE) -> io.NodeOutput:
        p = build_prompt(scene, lighting, camera, template)
        return io.NodeOutput(p, ui={"text": [p]})


# ----------------------------------------------------------------------------------------------- 7 tonemap preview
class MiniMaxH3_HDRTonemapPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRTonemapPreview",
            display_name="H3 HDR · Tonemap Preview",
            category=CATEGORY,
            description=("Display-only sRGB view of linear HDR: exposure, then a soft shoulder that rolls highlights off "
                         "instead of clipping them. For looking only - save the linear image, not this."),
            search_aliases=["hdr preview", "tonemap", "hdr view"],
            inputs=[
                io.Image.Input("image", tooltip="Scene-linear HDR."),
                io.Float.Input("exposure_ev", default=0.0, min=-16.0, max=16.0, step=0.1,
                               tooltip="Exposure in stops before the shoulder (-3 shows 3 stops more highlight)."),
                io.Float.Input("shoulder", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Highlight roll-off: 0 = hard clip at white, larger = softer, starts lower."),
            ],
            outputs=[io.Image.Output("image", tooltip="Display-referred sRGB 0..1 (exposure, shoulder, sRGB transfer) "
                                                      "for all frames; the node's own preview shows at most 16. For "
                                                      "viewing only - do not save it as HDR.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, exposure_ev=0.0, shoulder=0.5) -> io.NodeOutput:
        out = HM.tonemap_preview(_img(image), exposure_ev, shoulder).contiguous()
        n = out.shape[0]
        idx = sorted({round(i * (n - 1) / max(1, PREVIEW_MAX_FRAMES - 1)) for i in range(min(n, PREVIEW_MAX_FRAMES))})
        return io.NodeOutput(out, ui=ui.PreviewImage(out[idx], cls=cls))


# ----------------------------------------------------------------------------------------------- 8 load EXR sequence
def resolve_folder(folder: str) -> str:
    f = str(folder or "").strip().strip('"')
    if not f:
        raise ValueError("folder is empty: give the EXR folder (absolute, or relative to ComfyUI's input / output folder)")
    if os.path.isabs(f):
        if os.path.isdir(f):
            return f
        raise FileNotFoundError(f"EXR folder not found: {f}")
    import folder_paths
    for root in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        cand = os.path.join(root, f)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"EXR folder '{f}' is not in ComfyUI's input or output folder")


def exr_files(folder: str, start=0, count=0, step=1) -> list[str]:
    d = resolve_folder(folder)
    names = sorted((n for n in os.listdir(d) if n.lower().endswith(".exr")), key=_natural)
    if not names:
        raise FileNotFoundError(f"no .exr files in {d}")
    names = names[int(start)::max(1, int(step))]
    if int(count) > 0:
        names = names[:int(count)]
    if not names:
        raise ValueError(f"start {start} / step {step} select no frame of the {len(os.listdir(d))} files in {d}")
    return [os.path.join(d, n) for n in names]


class MiniMaxH3_HDRLoadEXRSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRLoadEXRSequence",
            has_intermediate_output=True,  # L9.290: core resends this ui when the node is served from cache
            display_name="H3 HDR · Load EXR Sequence",
            category=CATEGORY,
            description="Loads a folder of EXR frames (natural sort) as float IMAGE frames - no clamp, values above 1 kept.",
            search_aliases=["load exr", "exr sequence", "hdr load"],
            inputs=[
                io.String.Input("folder", default="", tooltip="Folder of .exr frames: absolute, or relative to "
                                                              "ComfyUI's input folder (then output folder)."),
                io.Int.Input("start", default=0, min=0, max=1000000, tooltip="First file (index in sorted order)."),
                io.Int.Input("count", default=0, min=0, max=100000, tooltip="Frames to load; 0 = all."),
                io.Int.Input("step", default=1, min=1, max=1000, tooltip="Every n-th file."),
            ],
            outputs=[io.Image.Output("images", tooltip="Float IMAGE [frames, H, W, 3] RGB as stored in the EXRs "
                                                       "(alpha dropped, grey -> 3 channels), not clamped."),
                     io.Int.Output("frame_count", tooltip="Number of EXR frames loaded after start / step / count."),
                     io.String.Output("report", tooltip="Text: folder, first and last file, size and min / max / % "
                                                        "above 1.")],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", start=0, count=0, step=1, **_):
        try:
            h = hashlib.sha256()
            for p in exr_files(folder, start, count, step):
                st = os.stat(p)
                h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
            return h.hexdigest()
        except Exception as exc:  # noqa: BLE001
            return f"error:{folder}:{exc}"

    @classmethod
    def execute(cls, folder, start=0, count=0, step=1) -> io.NodeOutput:
        files = exr_files(folder, start, count, step)
        frames = []
        for p in files:
            a = HIO.read_exr(p)
            if frames and a.shape != frames[0].shape:
                raise ValueError(f"{os.path.basename(p)} is {a.shape[1]}x{a.shape[0]}, the first frame "
                                 f"{frames[0].shape[1]}x{frames[0].shape[0]}: all frames must have one size")
            frames.append(a)
        img = torch.from_numpy(np.stack(frames)).float()
        report = (f"loaded {len(files)} EXR frames from {os.path.dirname(files[0])} ({os.path.basename(files[0])} .. "
                  f"{os.path.basename(files[-1])}), {img.shape[2]}x{img.shape[1]}; {_stats(img)}")
        return io.NodeOutput(img, len(files), report, ui={"text": [report]})


# ----------------------------------------------------------------------------------------------- 9 save EXR sequence
def save_exr_sequence(images, filename_prefix="hdr/exr/h3_hdr", precision="half", start_frame=1001, padding=4) -> str:
    lin = _img(images, "images")
    folder, name = _new_target(filename_prefix)
    big = HIO.HALF_MAX if precision == "half" else float(np.finfo(np.float32).max)
    nan_n = inf_n = sat_n = 0
    written, backend, vmax = [], "", float("-inf")
    for i in range(lin.shape[0]):
        a = lin[i].cpu().numpy().astype(np.float32)
        nan, inf = np.isnan(a), np.isinf(a)
        nan_n, inf_n = nan_n + int(nan.sum()), inf_n + int(inf.sum())
        a = np.nan_to_num(a, nan=0.0, posinf=big, neginf=-big)
        if precision == "half":
            sat = np.abs(a) > HIO.HALF_MAX
            sat_n += int(sat.sum())
            a = np.clip(a, -HIO.HALF_MAX, HIO.HALF_MAX)       # the half format's own limit, not a clamp to white
        p = os.path.join(folder, f"{name}.{int(start_frame) + i:0{int(padding)}d}.exr")
        backend = HIO.write_exr(p, a, precision)
        vmax = max(vmax, float(a.max()))
        written.append((p, a))
    for p, a in {0: written[0], 1: written[-1]}.values():           # read back first and last frame
        back = HIO.read_exr(p)
        ref = a.astype(np.float16).astype(np.float32) if precision == "half" else a
        if back.shape != ref.shape or not np.allclose(back, ref, rtol=1e-3, atol=1e-6):
            raise RuntimeError(f"read-back check failed for {p}: the file does not hold the values written")
    hdr = HIO.exr_header(written[0][0])
    notes = []
    if nan_n or inf_n:
        notes.append(f"{nan_n} NaN -> 0 and {inf_n} inf -> +/-{big:g}")
    if sat_n:
        notes.append(f"{sat_n} values beyond +/-65504 saturated (half-float maximum; use float32 to keep them)")
    return (f"wrote {len(written)} EXR frames ({precision}, ZIP, backend {backend}"
            + (f", header: {hdr['pixel_type']} / {hdr['compression']}" if hdr else "")
            + f") to {_rel(folder)}/{name}.{int(start_frame):0{int(padding)}d}.exr .. "
            f"{int(start_frame) + len(written) - 1:0{int(padding)}d}; max value {vmax:.4g}x white (not clamped); "
            f"read-back OK" + ("; " + "; ".join(notes) if notes else ""))


class MiniMaxH3_HDRSaveEXRSequence(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRSaveEXRSequence",
            display_name="H3 HDR · Save EXR Sequence",
            category=CATEGORY,
            description=("Saves linear HDR frames as an OpenEXR sequence (half or float32, ZIP) in a new folder under "
                         "ComfyUI's output folder. Never clamps light; reads the first and last frame back to check."),
            search_aliases=["save exr", "exr sequence", "hdr save"],
            inputs=[
                io.Image.Input("images", tooltip="Scene-linear HDR frames (Rec.709 unless you converted the gamut)."),
                io.String.Input("filename_prefix", default="hdr/exr/h3_hdr",
                                tooltip="sub/folder/name under output/; each run writes a new folder name_NNNNN/."),
                io.Combo.Input("precision", options=["half", "float32"], default="half",
                               tooltip="half = 16-bit float (to 65504, the usual VFX choice); float32 = full precision."),
                io.Int.Input("start_frame", default=1001, min=0, max=10000000, tooltip="Number of the first frame."),
                io.Int.Input("padding", default=4, min=1, max=8, tooltip="Digits of the frame number (4 -> 1001)."),
            ],
            outputs=[io.String.Output("report", tooltip="Text: EXR frames written, precision, backend, output file "
                                                        "range, max value, NaN / inf and half-float saturation counts, "
                                                        "read-back result.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix="hdr/exr/h3_hdr", precision="half", start_frame=1001,
                padding=4) -> io.NodeOutput:
        report = save_exr_sequence(images, filename_prefix, precision, start_frame, padding)
        return io.NodeOutput(report, ui={"text": [report]})


# --------------------------------------------------------------------------------------------------- 10 save log 16
def save_log16(images, filename_prefix="hdr/log16/h3_hdr_log", fmt="png", encode=NO_ENCODE, start_frame=1001,
               padding=4) -> str:
    x = _img(images, "images")
    code = x if encode == NO_ENCODE else HM.encode(x, encode)
    outside = float(((code < 0) | (code > 1)).any(-1).float().mean()) * 100
    folder, name = _new_target(filename_prefix)
    ext = "tif" if fmt == "tiff" else "png"
    for i in range(code.shape[0]):
        HIO.write_16bit(os.path.join(folder, f"{name}.{int(start_frame) + i:0{int(padding)}d}.{ext}"), code[i])
    return (f"wrote {code.shape[0]} 16-bit {ext.upper()} frames of the log image"
            + ("" if encode == NO_ENCODE else f" (encoded with {encode})")
            + f" to {_rel(folder)}/; {outside:.3f} % of pixels were outside 0..1 and held at the edge (log signal)")


class MiniMaxH3_HDRSaveLog16(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRSaveLog16",
            display_name="H3 HDR · Save Log 16-bit",
            category=CATEGORY,
            description=("Saves the LOG image (code values 0..1) as a 16-bit PNG or TIFF sequence - compact, lossless at "
                         "16 bits, decodes back to linear with Log Decode."),
            search_aliases=["save log", "16 bit png", "16 bit tiff"],
            inputs=[
                io.Image.Input("images", tooltip="Log code values (or linear, with 'encode' set)."),
                io.String.Input("filename_prefix", default="hdr/log16/h3_hdr_log",
                                tooltip="sub/folder/name under output/; each run writes a new folder name_NNNNN/."),
                io.Combo.Input("format", options=["png", "tiff"], default="png", tooltip="16-bit PNG or TIFF."),
                io.Combo.Input("encode", options=[NO_ENCODE] + HM.SUITE_ENCODINGS, default=NO_ENCODE,
                               tooltip="Input is already log -> 'none'; linear input -> pick the curve to encode with."),
                io.Int.Input("start_frame", default=1001, min=0, max=10000000, tooltip="Number of the first frame."),
                io.Int.Input("padding", default=4, min=1, max=8, tooltip="Digits of the frame number."),
            ],
            outputs=[io.String.Output("report", tooltip="Text: number of 16-bit PNG / TIFF frames written, the curve "
                                                        "used (if encoded), output folder and % of pixels held at 0 "
                                                        "or 1.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix="hdr/log16/h3_hdr_log", format="png", encode=NO_ENCODE,  # noqa: A002
                start_frame=1001, padding=4) -> io.NodeOutput:
        report = save_log16(images, filename_prefix, format, encode, start_frame, padding)
        return io.NodeOutput(report, ui={"text": [report]})


# ------------------------------------------------------------------------------------------------------ 11 HDR10
def save_hdr10(images, filename_prefix="hdr/hdr10/h3_hdr10", fps=24.0, crf=12, paper_white_nits=203.0,
               master_display_nits=1000) -> str:
    lin = _img(images, "images")
    path, _ = _new_target(filename_prefix, ".mp4")
    info = HIO.write_hdr10(path, lin, fps, crf, paper_white_nits, master_display_nits)
    tags = HIO.probe_video(path)
    bad = HIO.check_hdr10_tags(tags)
    if bad:
        raise RuntimeError(f"HDR10 check failed for {path}: " + "; ".join(bad))
    return (f"wrote {_rel(path)}: {info['frames']} frames {info['width']}x{info['height']}"
            + (" (padded to even size)" if info["padded"] else "")
            + f" @ {fps:g} fps, HEVC 10-bit 4:2:0 PQ / Rec.2020, CRF {crf}; 1.0 = {paper_white_nits:g} nits; "
            f"MaxCLL {info['max_cll']} / MaxFALL {info['max_fall']}; {info['clipped_fraction'] * 100:.3f} % of pixels "
            f"above 10000 nits clipped by PQ; {info['negative_fraction'] * 100:.3f} % out-of-gamut pixels zeroed; tags "
            f"verified ({tags['codec']}, {tags['pix_fmt']}, {tags['color_primaries']}, {tags['color_transfer']}, "
            f"{tags['color_space']})")


class MiniMaxH3_HDRSaveHDR10(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRSaveHDR10",
            display_name="H3 HDR · Save HDR10 Video",
            category=CATEGORY,
            description=("Linear Rec.709 HDR -> Rec.2020 -> nits (1.0 = paper white) -> PQ -> 10-bit HEVC .mp4 with HDR10 "
                         "metadata (ffmpeg / libx265). PQ holds up to 10000 nits; brighter is clipped and reported. The "
                         "file's colour tags are read back and checked."),
            search_aliases=["hdr10", "pq video", "save hdr video", "hevc hdr"],
            inputs=[
                io.Image.Input("images", tooltip="Scene-linear Rec.709 HDR frames."),
                io.String.Input("filename_prefix", default="hdr/hdr10/h3_hdr10",
                                tooltip="sub/folder/name under output/; each run writes name_NNNNN.mp4."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001,
                               tooltip="Frames per second of the written .mp4: passed to ffmpeg as the frame rate "
                                       "(-r) of the piped frames, so it sets playback speed and duration."),
                io.Int.Input("crf", default=12, min=0, max=51, tooltip="x265 quality: lower = better / larger."),
                io.Float.Input("paper_white_nits", default=203.0, min=50.0, max=1000.0, step=1.0,
                               tooltip="Brightness of diffuse white (1.0); 203 = BT.2408 reference white."),
                io.Int.Input("master_display_nits", default=1000, min=100, max=10000,
                             tooltip="Peak of the mastering display written in the HDR10 metadata."),
            ],
            outputs=[io.String.Output("report", tooltip="Text: written .mp4 path, frames, size, fps, CRF, paper white, "
                                                        "MaxCLL / MaxFALL, % clipped above 10000 nits, % out-of-gamut "
                                                        "zeroed and the verified colour tags.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix="hdr/hdr10/h3_hdr10", fps=24.0, crf=12, paper_white_nits=203.0,
                master_display_nits=1000) -> io.NodeOutput:
        report = save_hdr10(images, filename_prefix, fps, crf, paper_white_nits, master_display_nits)
        return io.NodeOutput(report, ui={"text": [report]})


# ------------------------------------------------------------------------------------------------- 17 eval report
def eval_report(decoded, ground_truth, encoding=HM.DEFAULT_ENCODING) -> str:
    rec, ref = _img(decoded, "decoded"), _img(ground_truth, "ground_truth")
    if rec.shape[1:] != ref.shape[1:]:
        raise ValueError(f"sizes differ: decoded {rec.shape[2]}x{rec.shape[1]}, ground truth {ref.shape[2]}x"
                         f"{ref.shape[1]} - resize one to the other first")
    n = min(rec.shape[0], ref.shape[0])
    rec, ref = rec[:n], ref[:n]
    ref_code, rec_code = HM.encode(ref, encoding), HM.encode(rec, encoding)
    base_code = HM.quantize8(ref_code)
    text = HM.format_report(encoding, ref, rec, ref_code, rec_code, HM.decode(base_code, encoding), base_code)
    head = f"frames compared: {n}" + (f" (decoded {decoded.shape[0]}, ground truth {ground_truth.shape[0]})"
                                       if decoded.shape[0] != ground_truth.shape[0] else "")
    return head + "\n" + text


class MiniMaxH3_HDREvalReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDREvalReport",
            display_name="H3 HDR · Eval Report",
            category=CATEGORY,
            description=("Decoded HDR vs ground truth: luminance error in stops per brightness band (mean, p99, bias), "
                         "worst channel, 8-bit rounding baseline and code-domain PSNR (the H3HDR-Test report)."),
            search_aliases=["hdr eval", "hdr report", "stop error"],
            inputs=[
                io.Image.Input("decoded", tooltip="Linear HDR from the pipeline (after Log Decode / Bias Correct)."),
                io.Image.Input("ground_truth", tooltip="The true linear HDR frames (e.g. Load EXR Sequence)."),
                io.Combo.Input("encoding", options=HM.SUITE_ENCODINGS, default=HM.DEFAULT_ENCODING,
                               tooltip="Curve for the code-domain PSNR and the 8-bit baseline."),
            ],
            outputs=[io.String.Output("report", tooltip="Text table: frames compared, luminance error in stops per "
                                                        "brightness band (mean, p99, bias), worst channel, 8-bit "
                                                        "rounding baseline and code-domain PSNR.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, decoded, ground_truth, encoding=HM.DEFAULT_ENCODING) -> io.NodeOutput:
        report = eval_report(decoded, ground_truth, encoding)
        return io.NodeOutput(report, ui={"text": [report]})


__all__ = [
    "MiniMaxH3_HDRLogEncode",
    "MiniMaxH3_HDRLogDecode",
    "MiniMaxH3_HDRBiasCorrect",
    "MiniMaxH3_HDRGamutConvert",
    "MiniMaxH3_HDRClipMask",
    "MiniMaxH3_HDRGainMapJoin",
    "MiniMaxH3_HDRPrompt",
    "MiniMaxH3_HDRTonemapPreview",
    "MiniMaxH3_HDRLoadEXRSequence",
    "MiniMaxH3_HDRSaveEXRSequence",
    "MiniMaxH3_HDRSaveLog16",
    "MiniMaxH3_HDRSaveHDR10",
    "MiniMaxH3_HDREvalReport",
]
