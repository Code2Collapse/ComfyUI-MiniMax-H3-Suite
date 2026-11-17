"""A29 HDR dataset nodes 12-15 (owner order A29, design docs/hdr/A29_DESIGN.md section 4): Fetch Poly Haven,
Fetch Hugging Face, HDRI -> Clips, Make Pairs. The work lives in mmx_utils/hdr_a29_dataset.py; it runs as a background
job (mmx_utils/hdr_a29_jobs.py) that the node either WAITS for (ComfyUI progress bar; ComfyUI's interrupt cancels it)
or leaves running (``wait`` off: the node returns at once, the jobs panel shows progress, log and Cancel).
Finished items are recorded in the work folder and skipped on a re-run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import hdr_a29_dataset as D  # noqa: E402
from mmx_utils import hdr_a29_jobs as J  # noqa: E402

CATEGORY = "MiniMax H3/HDR"
J.register_routes()

WAIT_TIP = ("On: the node waits for the job (progress bar; ComfyUI's Cancel stops it). Off: it returns at once and "
            "the job keeps running in the background - follow it in the HDR jobs panel on this node.")


def resolve_folder(path: str) -> str:
    """A relative folder lives under ComfyUI's output folder; an absolute one is used as given."""
    p = (path or "").strip().strip('"')
    if not p:
        raise ValueError("a folder is required")
    if os.path.isabs(p):
        return os.path.normpath(p)
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:  # noqa: BLE001 - outside a server (tests): relative to the working folder
        base = os.getcwd()
    return os.path.normpath(os.path.join(base, p))


def run_job(kind: str, title: str, key: str, fn, wait: bool):
    """Start ``fn(job)`` as a background job; wait for it (forwarding progress / interrupt) when asked.
    Returns (job, report text)."""
    job = J.start_thread(kind, title, fn, key=key)
    if not wait:
        return job, (f"{title}: started in the background (job {job.id}). Follow it in the HDR jobs panel; run the "
                     "node again when it has finished to get the final report (finished items are skipped).")
    progress, interrupted = J.comfy_waiter()
    try:
        J.wait_for(job, progress=progress, interrupted=interrupted)
    except J.JobCancelled:
        try:
            import comfy.model_management as mm
            mm.throw_exception_if_processing_interrupted()
        except ImportError:
            pass
        raise
    if job.status == "error":
        raise RuntimeError(f"{title} failed: {job.error}\n" + "\n".join(job.tail(15)))
    if job.status == "cancelled":
        raise RuntimeError(f"{title} was cancelled; run it again to continue (finished items are kept).")
    return job, json.dumps(job.result or {}, indent=1, default=str)


def _out(folder: str, job, report: str) -> io.NodeOutput:
    return io.NodeOutput(folder, report, ui={**J.ui_payload(job), "text": [report[:2000]]})


class MiniMaxH3_HDRFetchPolyHaven(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRFetchPolyHaven",
            display_name="H3 HDR Dataset · Fetch Poly Haven",
            category=CATEGORY,
            description=("Downloads CC0 HDRIs from Poly Haven (public API api.polyhaven.com) into a folder: most "
                         "downloaded first, resumable (.part files continue), size + md5 checked, files already there "
                         "are skipped. Writes a small .json per HDRI (name, categories, tags) used for captions."),
            search_aliases=["poly haven", "polyhaven hdri download", "hdr dataset fetch"],
            inputs=[
                io.String.Input("output_folder", default="hdr_dataset/hdri",
                                tooltip="Where the HDRIs go. Relative = inside ComfyUI's output folder."),
                io.Combo.Input("category", options=list(D.POLYHAVEN_CATEGORIES), default="all",
                               tooltip="Poly Haven HDRI category ('all' = no filter)."),
                io.Combo.Input("resolution", options=list(D.POLYHAVEN_RESOLUTIONS), default="2k",
                               tooltip="2k is plenty for 1344x768 renders at a 60 deg field of view; 8k files are "
                                       "large (100-300 MB each)."),
                io.Int.Input("limit", default=20, min=0, max=2000,
                             tooltip="How many HDRIs (0 = the whole category)."),
                io.Combo.Input("format", options=list(D.POLYHAVEN_FORMATS), default="hdr",
                               tooltip="hdr = Radiance (smaller); exr = OpenEXR."),
                io.Boolean.Input("wait", default=True, tooltip=WAIT_TIP),
            ],
            outputs=[io.String.Output("hdri_folder"), io.String.Output("report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, output_folder, category, resolution, limit, format, wait):  # noqa: A002
        try:
            return f"{category}|{resolution}|{limit}|{format}|" + J.folder_signature(resolve_folder(output_folder),
                                                                                      D.FETCH_STATE)
        except Exception:  # noqa: BLE001
            return f"{output_folder}|{category}|{resolution}|{limit}|{format}"

    @classmethod
    def execute(cls, output_folder, category, resolution, limit, format, wait) -> io.NodeOutput:  # noqa: A002
        folder = resolve_folder(output_folder)
        job, rep = run_job("fetch", f"Poly Haven {category} {resolution} x{limit or 'all'}", f"fetch:{folder}",
                           lambda job: D.fetch_polyhaven(folder, category, resolution, limit, format, job=job), wait)
        return _out(folder, job, rep)


class MiniMaxH3_HDRFetchHuggingFace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRFetchHuggingFace",
            display_name="H3 HDR Dataset · Fetch Hugging Face",
            category=CATEGORY,
            description=("Downloads the files of a Hugging Face repo that match the allow patterns (snapshot "
                         "download). The total size is checked BEFORE anything downloads and the fetch is refused "
                         "above the GB cap. Private / gated repos use the HF_TOKEN environment variable (never an "
                         "input, never logged)."),
            search_aliases=["hugging face dataset download", "hf hdr dataset"],
            inputs=[
                io.String.Input("repo_id", default="", tooltip="owner/name of the repo."),
                io.Combo.Input("repo_type", options=["dataset", "model"], default="dataset"),
                io.String.Input("allow_patterns", default="*.exr, *.hdr",
                                tooltip="Comma / newline separated glob patterns (e.g. 'hdri/*.exr')."),
                io.Float.Input("cap_gb", default=5.0, min=0.01, max=10000.0, step=0.5,
                               tooltip="Refuse the fetch when the matching files total more than this."),
                io.String.Input("output_folder", default="hdr_dataset/hf",
                                tooltip="Relative = inside ComfyUI's output folder."),
                io.String.Input("revision", default="", tooltip="Branch / tag / commit; empty = main."),
                io.Boolean.Input("wait", default=True, tooltip=WAIT_TIP),
            ],
            outputs=[io.String.Output("folder"), io.String.Output("report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, repo_id, repo_type, allow_patterns, cap_gb, output_folder, revision, wait):
        try:
            sig = J.folder_signature(resolve_folder(output_folder))
        except Exception:  # noqa: BLE001
            sig = ""
        return f"{repo_id}|{repo_type}|{allow_patterns}|{cap_gb}|{revision}|{sig}"

    @classmethod
    def execute(cls, repo_id, repo_type, allow_patterns, cap_gb, output_folder, revision, wait) -> io.NodeOutput:
        folder = resolve_folder(output_folder)
        job, rep = run_job("fetch", f"Hugging Face {repo_id}", f"fetch:{folder}",
                           lambda job: D.fetch_huggingface(repo_id, folder, allow_patterns, cap_gb,
                                                           repo_type=repo_type, revision=revision, job=job), wait)
        return _out(folder, job, rep)


class MiniMaxH3_HDRHDRIToClips(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRHDRIToClips",
            display_name="H3 HDR Dataset · HDRI -> Clips",
            category=CATEGORY,
            description=("Renders camera moves (pan / tilt / push-in, seeded) through each equirect HDRI (.hdr / "
                         ".exr) into EXR (half) frame folders at the H3 draft size, scene-linear Rec.709 with 1.0 = "
                         "diffuse white (exposure: the clip's median becomes 18 % grey). Frame counts on H3's 17k+5 "
                         "grid. Finished clips are skipped."),
            search_aliases=["hdri to video", "equirect render", "hdr dataset clips"],
            inputs=[
                io.String.Input("hdri_folder", default="hdr_dataset/hdri",
                                tooltip="Folder of .hdr / .exr panoramas (Fetch Poly Haven's output)."),
                io.String.Input("output_folder", default="hdr_dataset/clips",
                                tooltip="One sub-folder per clip. Relative = inside ComfyUI's output folder."),
                io.Int.Input("clips_per_hdri", default=2, min=1, max=64),
                io.Int.Input("frames", default=D.DEFAULT_FRAMES, min=22, max=1024,
                             tooltip="17k+5: 22, 39, 56, 73 (3 s at 24 fps), 90, 107, 124 ..."),
                io.Int.Input("width", default=D.DRAFT_W, min=256, max=4096, step=32),
                io.Int.Input("height", default=D.DRAFT_H, min=256, max=4096, step=32),
                io.Float.Input("fov_deg", default=60.0, min=20.0, max=120.0, step=1.0,
                               tooltip="Horizontal field of view at the start of the move."),
                io.Combo.Input("motion", options=list(D.MOTIONS), default="mix",
                               tooltip="mix = a seeded choice of pan / tilt / push per clip."),
                io.Combo.Input("exposure", options=list(D.EXPOSURES), default="grey_0.18"),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1),
                io.Int.Input("limit", default=0, min=0, max=100000, tooltip="Use only the first N HDRIs (0 = all)."),
                io.Boolean.Input("wait", default=True, tooltip=WAIT_TIP),
            ],
            outputs=[io.String.Output("clips_folder"), io.String.Output("report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, hdri_folder, output_folder, clips_per_hdri, frames, width, height, fov_deg, motion,
                           exposure, seed, limit, wait):
        try:
            sig = J.folder_signature(resolve_folder(output_folder), D.CLIPS_STATE) + "|" + \
                J.folder_signature(resolve_folder(hdri_folder))
        except Exception:  # noqa: BLE001
            sig = ""
        return f"{hdri_folder}|{clips_per_hdri}|{frames}|{width}|{height}|{fov_deg}|{motion}|{exposure}|{seed}|" \
               f"{limit}|{sig}"

    @classmethod
    def execute(cls, hdri_folder, output_folder, clips_per_hdri, frames, width, height, fov_deg, motion, exposure,
                seed, limit, wait) -> io.NodeOutput:
        D.check_frames(frames)                                   # refuse before a job starts
        D.check_side(width, "width")
        D.check_side(height, "height")
        src, folder = resolve_folder(hdri_folder), resolve_folder(output_folder)
        job, rep = run_job("clips", f"HDRI -> clips ({Path(src).name})", f"clips:{folder}",
                           lambda job: D.hdri_to_clips(src, folder, clips_per_hdri=clips_per_hdri, frames=frames,
                                                       width=width, height=height, fov_deg=fov_deg, motion=motion,
                                                       exposure=exposure, seed=seed, limit=limit, job=job), wait)
        return _out(folder, job, rep)


class MiniMaxH3_HDRMakePairs(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRMakePairs",
            display_name="H3 HDR Dataset · Make Pairs",
            category=CATEGORY,
            description=("EXR clips -> the HDR IC-LoRA training set ai-toolkit reads: control/<clip>.mp4 = SDR "
                         "reference (random exposure shift, half hard clip / half soft roll-off, a share left "
                         "unclipped, sRGB 8-bit h264 crf 16-28) and targets/<clip>.mp4 = the same frames in "
                         "diffhdr_log_16384 as lossless RGB (8-bit codes), + targets/<clip>.txt caption and "
                         "meta.jsonl with every parameter. Same file names in both folders (that is how the trainer "
                         "pairs them). Finished pairs are skipped."),
            search_aliases=["hdr pairs", "sdr hdr pairs", "hdr dataset pairs", "ic lora dataset"],
            inputs=[
                io.String.Input("clips_folder", default="hdr_dataset/clips",
                                tooltip="HDRI -> Clips output (one folder per clip)."),
                io.String.Input("output_folder", default="hdr_dataset/pairs",
                                tooltip="Gets targets/, control/ and meta.jsonl. Relative = inside ComfyUI's output."),
                io.Int.Input("seed", default=0, min=0, max=2**31 - 1),
                io.Float.Input("ev_range", default=1.5, min=0.0, max=6.0, step=0.1,
                               tooltip="Control exposure shift drawn from -ev_range..+ev_range stops per clip."),
                io.Float.Input("unclipped_share", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of clips whose SDR control keeps every highlight (order: 15 %)."),
                io.Int.Input("crf_min", default=16, min=0, max=51),
                io.Int.Input("crf_max", default=28, min=0, max=51),
                io.String.Input("caption_template", default=D.DEFAULT_CAPTION_TEMPLATE, multiline=True,
                                tooltip="HDR Prompt template; {scene}, {lighting} and {camera} are filled per clip "
                                        "from the HDRI name / Poly Haven categories and the camera move."),
                io.Int.Input("fps", default=24, min=1, max=120, tooltip="H3 trains at 24 fps."),
                io.Boolean.Input("wait", default=True, tooltip=WAIT_TIP),
            ],
            outputs=[io.String.Output("dataset_folder"), io.String.Output("report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, clips_folder, output_folder, seed, ev_range, unclipped_share, crf_min, crf_max,
                           caption_template, fps, wait):
        try:
            sig = J.folder_signature(resolve_folder(output_folder), D.PAIRS_STATE) + "|" + \
                J.folder_signature(resolve_folder(clips_folder))
        except Exception:  # noqa: BLE001
            sig = ""
        return f"{clips_folder}|{seed}|{ev_range}|{unclipped_share}|{crf_min}|{crf_max}|{caption_template}|{fps}|{sig}"

    @classmethod
    def execute(cls, clips_folder, output_folder, seed, ev_range, unclipped_share, crf_min, crf_max,
                caption_template, fps, wait) -> io.NodeOutput:
        if crf_min > crf_max:
            raise ValueError("crf_min must not be above crf_max")
        D.render_caption(caption_template, {"scene": "s", "lighting": "l", "camera": "c"})   # template check first
        src, folder = resolve_folder(clips_folder), resolve_folder(output_folder)
        job, rep = run_job("pairs", f"Make pairs ({Path(src).name})", f"pairs:{folder}",
                           lambda job: D.make_pairs(src, folder, seed=seed, ev_range=ev_range,
                                                    unclipped_share=unclipped_share, crf_min=crf_min,
                                                    crf_max=crf_max, caption_template=caption_template, fps=fps,
                                                    job=job), wait)
        return _out(folder, job, rep)


NODES = [MiniMaxH3_HDRFetchPolyHaven, MiniMaxH3_HDRFetchHuggingFace, MiniMaxH3_HDRHDRIToClips, MiniMaxH3_HDRMakePairs]
