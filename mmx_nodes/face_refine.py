# MIT License — ComfyUI-H3-FaceRefine
# Author: Carasibana
# PORTED FROM: ComfyUI-H3-FaceRefine :: nodes.py @ HEAD (5 nodes; H3PerFrameDenoise deduped)

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils.av_latent import assert_nested_samples, inject_video_stream, pack_latent_dict
from mmx_utils.face_refine import (
    detector_list,
    format_transform_info,
    run_sam_face_masks,
    stitch_refined_faces,
    track_face_crop,
)
from mmx_utils.transform_types import H3Transform, H3TransformType

try:
    import comfy.model_management as mm
    import comfy.nested_tensor as nested_tensor
except Exception:  # not just ImportError: the comfy_kitchen skew raises AttributeError
    mm = None
    nested_tensor = None

CATEGORY = "MiniMax H3/Face"


def _detector_options() -> list[str]:
    return detector_list() or ["face_yolov8m.pt"]


def _fallback_options() -> list[str]:
    return ["none"] + _detector_options()


class MiniMaxH3_FaceTrackCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        dets = _detector_options()
        return io.Schema(
            node_id="MiniMaxH3_FaceTrackCrop",
            display_name="H3 Face Track + Crop",
            category=CATEGORY,
            description=(
                "Per-frame face track -> smoothed, normalised crop -> constant-size batch for H3, "
                "plus the transform needed to paste the result back."
            ),
            inputs=[
                io.Image.Input("images",
                               tooltip="Source video frames (IMAGE batch) to track the face in and crop from."),
                io.Combo.Input("detector", options=dets, default=dets[0],
                               tooltip="Ultralytics YOLO face detector from models/ultralytics (bbox) that finds "
                                       "the face box in every frame. Loaded once and cached."),
                io.Float.Input("confidence", default=0.35, min=0.05, max=0.95, step=0.05,
                               tooltip="Minimum detector score (0-1) for a box to count, for the face detector and "
                                       "the fallback detector. Lower finds more faces but more false boxes; frames "
                                       "with no face are interpolated."),
                io.Float.Input(
                    "crop_factor",
                    default=2.5,
                    min=1.2,
                    max=8.0,
                    step=0.1,
                    tooltip=(
                        "Crop side as a multiple of detected face HEIGHT. 2.5 puts the "
                        "face at ~40% of the crop, comfortably inside H3's good regime. "
                        "Bigger = more context so the seam lands in hair/background, but "
                        "less magnification. 2.0-3.0 is the useful range."
                    ),
                ),
                io.Int.Input(
                    "canvas_width",
                    default=512,
                    min=128,
                    max=1344,
                    step=32,
                    tooltip=(
                        "Resolution H3 generates at. 512 is cheap; 768 is H3's native "
                        "short edge and gives the best faces. Ignored when canvas_mode "
                        "is not 'manual'. Cost scales with area: 768 is 2.25x the "
                        "latent tokens of 512."
                    ), extra_dict={"c2c_show_if": {"canvas_mode": ["manual"]}},
                ),
                io.Int.Input("canvas_height", default=512, min=128, max=1344, step=32,
                             tooltip="Height in pixels of every crop (the H3 canvas). With canvas_width it sets "
                                     "the crop aspect. Ignored when canvas_mode is not 'manual' (the auto modes "
                                     "use a square canvas).",
                             extra_dict={"c2c_show_if": {"canvas_mode": ["manual"]}}),
                io.Combo.Input(
                    "canvas_mode",
                    options=["manual", "auto_no_downscale", "auto_capped_768"],
                    default="manual",
                    tooltip=(
                        "manual: use canvas_width/height as given.\n"
                        "auto_no_downscale: size the canvas from the LARGEST crop in "
                        "the clip so no frame is ever downscaled (magnification never "
                        "drops below 1.0x). Can get expensive on clips that include "
                        "close-ups.\n"
                        "auto_capped_768: same, but clamped to 768 - H3's native short "
                        "edge and a sane VRAM ceiling."
                    ),
                ),
                io.Int.Input(
                    "smooth_window",
                    default=21,
                    min=1,
                    max=201,
                    step=2,
                    tooltip=(
                        "Frames of smoothing on the crop CENTRE. 21 at 24fps is ~0.9s. "
                        "Raise if the box still shivers; lower if it lags behind fast "
                        "head movement."
                    ), extra_dict={"c2c_section": "Smoothing"},
                ),
                io.Int.Input(
                    "size_smooth_window",
                    default=51,
                    min=1,
                    max=201,
                    step=2,
                    tooltip=(
                        "Frames of smoothing on the crop SIZE. Wants MORE than the "
                        "centre: size jitter makes the crop breathe, which changes the "
                        "resample factor every frame and reads as shimmer. Real zoom "
                        "moves are slow, so heavy smoothing here costs nothing."
                    ), extra_dict={"c2c_section": "Smoothing"},
                ),
                io.Combo.Input(
                    "smooth_method",
                    options=["gaussian", "savgol", "moving_average"],
                    default="gaussian",
                    tooltip=(
                        "gaussian: best jitter rejection. savgol: preserves the shape of "
                        "a push-in better at large windows. moving_average: the old "
                        "boxcar, leaves residual jitter."
                    ), extra_dict={"c2c_section": "Smoothing"},
                ),
                io.Combo.Input(
                    "size_mode",
                    options=["max_of_clip", "per_frame"],
                    default="per_frame",
                    tooltip=(
                        "per_frame: constant face-fraction in every crop (correct for "
                        "push-ins). max_of_clip: one size for the whole clip, only "
                        "useful when the shot is genuinely static."
                    ),
                ),
                io.Image.Input(
                    "identity_reference",
                    optional=True,
                    tooltip=(
                        "A clear face image of the person to track. When supplied, the "
                        "subject is chosen by FACE IDENTITY rather than by size, so a "
                        "crowd scene locks onto the right person even when someone else "
                        "is briefly larger or nearer.\n\n"
                        "Without it, 'largest' has no notion of WHO it is following - "
                        "it just takes the biggest box each frame, which switches "
                        "subject whenever the framing changes.\n\n"
                        "FOR MULTIPLE PEOPLE: run the pipeline once per subject, each "
                        "with that person's reference here and their own refs on the H3 "
                        "node, and chain them - feed run 1's stitched output in as run "
                        "2's base_images. The composites accumulate."
                    ),
                ),
                io.Boolean.Input(
                    "identity_track",
                    default=True,
                    tooltip=(
                        "Hold one subject through a crowd. Continuity (nearest box to "
                        "the previous position) decides most frames; the face-identity "
                        "embedding is consulted only when two candidates are similarly "
                        "plausible or their boxes overlap - which is both the accurate "
                        "and the cheap arrangement, since the embedding model then runs "
                        "on a handful of frames instead of all of them. "
                        "The anchor is taken FROM THE CLIP by default (frames where one "
                        "face clearly dominates), because an external stylised reference "
                        "sits in a different domain - measured similarity between an "
                        "illustration and a render of the same character was only 0.305, "
                        "where same-domain faces score 0.5-0.7."
                    ), extra_dict={"c2c_section": "Identity"},
                ),
                io.Float.Input(
                    "identity_threshold",
                    default=0.28,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Minimum cosine similarity to accept a face as the reference "
                        "person. Below this the frame falls back to continuity (nearest "
                        "to the previous position at a similar size), which is what "
                        "carries tracking through profiles and partial occlusion where "
                        "embeddings become unreliable."
                    ), extra_dict={"c2c_section": "Identity", "c2c_show_if": {"identity_track": [True]}},
                ),
                io.Combo.Input(
                    "select",
                    options=["largest", "most_central"],
                    default="largest",
                    tooltip=(
                        "Used only when no identity_reference is connected, and as the "
                        "first-frame tie-break."
                    ), extra_dict={"c2c_section": "Identity"},
                ),
                io.Combo.Input(
                    "fallback_detector",
                    options=_fallback_options(),
                    default="none",
                    tooltip=(
                        "Used only on frames where the FACE detector finds nothing "
                        "(subject turned away). A person/body model such as "
                        "segm\\person_yolov8m-seg.pt gives a real head position from the "
                        "top of the body box, which beats interpolating blindly between "
                        "the last and next face. Set 'none' to interpolate instead."
                    ), extra_dict={"c2c_section": "Fallback"},
                ),
                io.Float.Input(
                    "fallback_head_frac",
                    default=0.5,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    tooltip=(
                        "Head centre as a multiple of face height below the top of the "
                        "person box. 0.5 puts it half a face-height down, which is about "
                        "right for a head seen from behind."
                    ), extra_dict={"c2c_section": "Fallback", "c2c_show_if": {"fallback_detector": {"not": ["none"]}}},
                ),
            ],
            outputs=[
                io.Image.Output("crops",
                                tooltip="IMAGE batch, one canvas_w×canvas_h face crop per input frame (smoothed "
                                        "track). Feed to the H3 refine pass and to H3 Face Mask (SAM)."),
                H3TransformType.Output("transform",
                                       tooltip="Per-frame crop boxes, canvas and source size, detection flags and "
                                               "fade weights. Needed by H3 Face Stitch Back, Face Mask (SAM) and "
                                               "Face Transform Info."),
                io.Image.Output("preview",
                                tooltip="Source frames with the crop box drawn: green = face detected, yellow = body "
                                        "fallback, red = interpolated across a gap."),
                io.String.Output("report",
                                 tooltip="Text summary: tracking/identity counts, detection rate, face and crop "
                                         "sizes, magnification, jitter before/after smoothing, dropouts and warnings."),
                io.Int.Output("canvas_w",
                              tooltip="Final crop width in pixels (after canvas_mode). Wire to the H3 node's width."),
                io.Int.Output("canvas_h",
                              tooltip="Final crop height in pixels (after canvas_mode). Wire to the H3 node's "
                                      "height."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        detector,
        confidence,
        crop_factor,
        canvas_width,
        canvas_height,
        canvas_mode,
        smooth_window,
        size_smooth_window,
        smooth_method,
        size_mode,
        identity_reference=None,
        identity_track=True,
        identity_threshold=0.28,
        select="largest",
        fallback_detector="none",
        fallback_head_frac=0.5,
    ) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)

        def _interrupt():
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()

        crops, transform, preview, report, cw, ch = track_face_crop(
            images,
            detector=detector,
            confidence=float(confidence),
            crop_factor=float(crop_factor),
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            canvas_mode=canvas_mode,
            smooth_window=int(smooth_window),
            size_smooth_window=int(size_smooth_window),
            smooth_method=smooth_method,
            size_mode=size_mode,
            select=select,
            fallback_detector=fallback_detector,
            fallback_head_frac=float(fallback_head_frac),
            identity_reference=identity_reference,
            identity_threshold=float(identity_threshold),
            identity_track=bool(identity_track),
            interrupt_check=_interrupt,
        )
        return io.NodeOutput(crops, transform, preview, report, cw, ch)


class MiniMaxH3_FaceStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FaceStitch",
            display_name="H3 Face Stitch Back",
            category=CATEGORY,
            description="Composite H3-refined face crops back into the source frames.",
            inputs=[
                io.Image.Input("base_images",
                               tooltip="The full frames the crops were taken from (same size as the transform's "
                                       "source, else an error). The refined faces are pasted into a copy of these."),
                io.Image.Input("refined_crops",
                               tooltip="H3-refined face crops, one per frame, on the crop canvas. If the frame count "
                                       "differs from base_images, only the shorter count is pasted."),
                H3TransformType.Input("transform",
                                      tooltip="The transform output of H3 Face Track + Crop for this clip: maps each "
                                              "crop back to its box in the frame and supplies detection fade weights."),
                io.Combo.Input(
                    "paste_region",
                    options=["face_only", "face_ellipse", "full_crop"],
                    default="face_only",
                    tooltip=(
                        "WHAT gets composited back. face_only / face_ellipse paste just "
                        "the detected face box (FaceDetailer's behaviour - the wider "
                        "crop exists to give the sampler context, not to be pasted). "
                        "full_crop pastes the whole crop including hair, shoulders and "
                        "background, which risks a visible rectangle if H3 alters them."
                    ), extra_dict={"c2c_hide_if_linked": ["masks"]},
                ),
                io.Int.Input(
                    "mask_dilation",
                    default=16,
                    min=0,
                    max=256,
                    step=2,
                    tooltip=(
                        "Grow the face box before blurring, in canvas px. Impact Pack "
                        "dilates the same way so the blur has room and the blend does "
                        "not eat into the face itself."
                    ),
                ),
                io.Int.Input(
                    "feather",
                    default=6,
                    min=0,
                    max=256,
                    step=2,
                    tooltip=(
                        "Gaussian blur radius on the paste mask, in SOURCE pixels. "
                        "Measured against the final frame, not the canvas, so the blend "
                        "is the same physical width whatever this frame's magnification "
                        "happens to be.\n\n"
                        "Canvas-relative feather is a trap: a 75px crop blown up to 512 "
                        "makes a 40px canvas feather only ~6 source px, while a 720px "
                        "crop makes it ~56. The blend ends up TIGHTEST exactly where the "
                        "face is smallest and the composite needs the most help - which "
                        "reads as a hard edge appearing as a shot zooms out."
                    ),
                ),
                io.Float.Input(
                    "colour_match",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Match the refined crop's per-channel mean/std to the region it "
                        "replaces. The crop and the full frame went through independent "
                        "passes, so without this the face can come back subtly brighter "
                        "or differently tinted and read as pasted on."
                    ), extra_dict={"c2c_section": "Blend"},
                ),
                io.Float.Input(
                    "blend",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Global opacity of the refined face. Below 1.0 mixes back toward "
                        "the original - useful to dial back over-sharpening."
                    ), extra_dict={"c2c_section": "Blend"},
                ),
                io.Combo.Input(
                    "undetected_frames",
                    options=["fade_out", "skip", "composite_anyway"],
                    default="fade_out",
                    tooltip=(
                        "What to do on frames where no FACE was found (turned away / "
                        "occluded). ALL frames are still sent through H3 either way - "
                        "that is what keeps it temporally consistent - this only "
                        "controls whether the result is pasted back.\n"
                        "fade_out: ramp the composite to zero across the gap (smooth, "
                        "no pop, recommended).\n"
                        "skip: hard cut - those frames keep original pixels exactly.\n"
                        "composite_anyway: paste regardless. Risks H3 hallucinating a "
                        "face onto the back of a head."
                    ), extra_dict={"c2c_section": "Blend"},
                ),
                io.Boolean.Input(
                    "feather_scales_with_crop",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Old behaviour: treat feather as CANVAS pixels, so the blend "
                        "narrows as the crop shrinks. Leave off."
                    ), extra_dict={"c2c_hidden": True},
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip=(
                        "Optional per-frame paste masks in CANVAS space, e.g. from "
                        "H3 Face Mask (SAM). Overrides paste_region. This is the "
                        "FaceDetailer bbox+SAM path: the mask follows the actual face so "
                        "the blend falls on the jaw and hairline instead of an arbitrary "
                        "rectangle. With a SAM mask use a SMALL feather (4-8); a "
                        "rectangle needs much more."
                    ),
                ),
            ],
            outputs=[io.Image.Output("images",
                                     tooltip="IMAGE batch: base_images with the refined faces composited in "
                                             "(same size and frame count as base_images).")],
        )

    @classmethod
    def execute(
        cls,
        base_images,
        refined_crops,
        transform: H3Transform,
        paste_region="face_only",
        mask_dilation=16,
        feather=6,
        colour_match=1.0,
        blend=1.0,
        undetected_frames="fade_out",
        feather_scales_with_crop=False,
        masks=None,
    ) -> io.NodeOutput:
        if base_images.ndim == 3:
            base_images = base_images.unsqueeze(0)
        if refined_crops.ndim == 3:
            refined_crops = refined_crops.unsqueeze(0)

        dev = base_images.device
        if mm is not None:
            try:
                dev = mm.get_torch_device()
            except Exception:
                pass

        def _interrupt():
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()

        out = stitch_refined_faces(
            base_images,
            refined_crops,
            transform,
            paste_region=paste_region,
            mask_dilation=int(mask_dilation),
            feather=int(feather),
            colour_match=float(colour_match),
            blend=float(blend),
            undetected_frames=undetected_frames,
            masks=masks,
            feather_scales_with_crop=bool(feather_scales_with_crop),
            interrupt_check=_interrupt,
            device=dev,
        )
        return io.NodeOutput(out)


class MiniMaxH3_InjectVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_InjectVideoLatent",
            display_name="H3 Inject Video Latent (img2img)",
            category=CATEGORY,
            description=(
                "Encode real frames into the video stream of an H3 joint AV latent. "
                "Pair with MiniMaxH3_NativeAudioLock for the audio stream, and set strength with "
                "BasicScheduler's denoise - NOT with SplitSigmas."
            ),
            inputs=[
                io.Latent.Input("av_latent",
                                tooltip="H3 joint audio-video LATENT (video + audio streams) whose video stream is "
                                        "replaced. Sets the target latent size; its audio stream is kept as is."),
                io.Image.Input("images",
                               tooltip="Frames to encode into the video stream. Width/height must encode to the "
                                       "latent's size (pixels/16), else an error; a frame count that encodes to a "
                                       "different latent length is trimmed or padded, with a warning."),
                io.Vae.Input("vae",
                             tooltip="H3 video VAE used to encode the frames."),
            ],
            outputs=[
                io.Latent.Output("av_latent",
                                 tooltip="Copy of the input LATENT with the encoded frames as its video stream. Sample "
                                         "it with denoise below 1 (BasicScheduler) for img2img."),
                io.String.Output("report",
                                 tooltip="Text summary: encoded latent shape, stream count, input frame count and "
                                         "size, and any temporal trim/pad warning."),
            ],
        )

    @classmethod
    def execute(cls, av_latent, images, vae) -> io.NodeOutput:
        members = assert_nested_samples(av_latent.get("samples"))
        video_tmpl = members[0]

        encoded = vae.encode(images[..., :3])
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(0).movedim(1, 2)

        members, note = inject_video_stream(members, encoded, video_tmpl)
        if nested_tensor is None:
            raise RuntimeError("MiniMaxH3_InjectVideoLatent requires comfy.nested_tensor.")
        out = pack_latent_dict(av_latent, members, None, nested_tensor.NestedTensor)

        enc = members[0]
        report = (
            f"injected video latent {tuple(enc.shape)} into AV latent "
            f"(streams={len(members)})\n"
            f"{note + chr(10) if note else ''}"
            f"frames_in={images.shape[0]}  {images.shape[2]}x{images.shape[1]}px"
        )
        return io.NodeOutput(out, report)


class MiniMaxH3_FaceMaskSAM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FaceMaskSAM",
            display_name="H3 Face Mask (SAM)",
            category=CATEGORY,
            description="Per-frame SAM face masks on the stabilised crops, temporally smoothed.",
            inputs=[
                io.Image.Input(
                    "crops",
                    tooltip=(
                        "Wire the INPUT crops here - the 'crops' output of H3 Face "
                        "Track + Crop - NOT the refined/decoded result.\n\n"
                        "This matches FaceDetailer: make_sam_mask() runs on the SOURCE "
                        "image and the resulting mask is what the enhanced patch is "
                        "later pasted through. Generation never feeds back into the "
                        "mask.\n\n"
                        "Masking the generated result instead is actively wrong: if the "
                        "model nudges the face inward, the mask traces the NEW, smaller "
                        "silhouette and the ORIGINAL face pokes out past it - most "
                        "visibly the nose on profile shots. Masking the input covers "
                        "where the face actually is in the footage being replaced.\n\n"
                        "It is also cheaper: no dependency on the sampler, so SAM need "
                        "not be resident alongside the video model."
                    ),
                ),
                io.Custom("SAM_MODEL").Input("sam_model",
                                             tooltip="SAM model (Impact Pack SAMLoader). Prompted per frame with the "
                                                     "face box and its centre point; moved to the device for the run "
                                                     "and released afterwards."),
                H3TransformType.Input("transform",
                                      tooltip="The transform from H3 Face Track + Crop; its per-frame face rectangles "
                                              "(canvas space) are the SAM box prompts and the fallback masks."),
                io.Float.Input("threshold", default=0.93, min=0.0, max=1.0, step=0.01,
                               tooltip="Passed to the SAM model's predict() as its score threshold (0-1). Higher "
                                       "rejects more masks; frames with no accepted mask use sam_fallback."),
                io.Int.Input(
                    "dilation",
                    default=0,
                    min=0,
                    max=128,
                    step=2,
                    tooltip=(
                        "Mirrors FaceDetailer's sam_dilation default of 0. "
                        "SAM masks are accurate, so they rarely need growing."
                    ), extra_dict={"c2c_section": "Mask"},
                ),
                io.Int.Input(
                    "temporal_smooth",
                    default=5,
                    min=1,
                    max=31,
                    step=2,
                    tooltip=(
                        "Frames of averaging across the mask stack. 1 disables it and "
                        "you will likely see the mask edge shimmer."
                    ), extra_dict={"c2c_section": "Mask"},
                ),
                # appended last (T7): the default keeps saved workflows' rectangle
                io.Combo.Input(
                    "sam_fallback",
                    options=["face rect", "hold nearest SAM mask (experimental)"],
                    default="face rect",
                    optional=True,
                    tooltip=(
                        "A frame SAM cannot segment: the face rectangle (a box edge can pop in a feathered paste; "
                        "the report names those frames), or the nearest segmented frame's mask moved with the face."
                    ), extra_dict={"c2c_section": "Mask"},
                ),
            ],
            outputs=[
                io.Mask.Output("masks",
                               tooltip="MASK batch in crop-canvas space, one per frame (1 = face), dilated and "
                                       "temporally averaged. Wire to H3 Face Stitch Back masks."),
                io.String.Output("report",
                                 tooltip="Text summary: frames segmented by SAM, frames that fell back (rectangle or "
                                         "held mask), dilation, smoothing and mean mask coverage."),
            ],
        )

    @classmethod
    def execute(
        cls,
        crops,
        sam_model,
        transform: H3Transform,
        threshold=0.93,
        dilation=0,
        temporal_smooth=5,
        sam_fallback="face rect",
    ) -> io.NodeOutput:
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)

        def _interrupt():
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()

        progress = None
        try:
            import comfy.utils as cu

            pbar = cu.ProgressBar(crops.shape[0])

            def progress(_i):
                pbar.update(1)

        except Exception:
            progress = None

        masks, report = run_sam_face_masks(
            crops,
            sam_model,
            transform,
            threshold=float(threshold),
            dilation=int(dilation),
            temporal_smooth=int(temporal_smooth),
            interrupt_check=_interrupt,
            progress_cb=progress,
            fallback=str(sam_fallback),
        )
        return io.NodeOutput(masks, report)


class MiniMaxH3_FaceTransformInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_FaceTransformInfo",
            display_name="H3 Face Transform Info",
            category=CATEGORY,
            description="Print the per-frame transform - sanity-check tracking before spending GPU time.",
            inputs=[
                H3TransformType.Input("transform",
                                      tooltip="The transform from H3 Face Track + Crop to print."),
                io.Int.Input("max_rows", default=12, min=1, max=400,
                             tooltip="Roughly how many frame rows to list: every (frames / max_rows)-th frame is "
                                     "shown."),
            ],
            outputs=[io.String.Output("info",
                                      tooltip="Text table: frame count, canvas and source size, then per sampled "
                                              "frame the crop box x/y/w/h in source pixels and its magnification. "
                                              "Also printed to the console.")],
        )

    @classmethod
    def execute(cls, transform: H3Transform, max_rows=12) -> io.NodeOutput:
        txt = format_transform_info(transform, int(max_rows))
        print("[MiniMaxH3_FaceTransformInfo]\n" + txt)
        return io.NodeOutput(txt)
