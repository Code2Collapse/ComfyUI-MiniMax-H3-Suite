"""A29 node 16: HDR Train IC-LoRA (owner order A29, design docs/hdr/A29_DESIGN.md sections 1 and 4).

Writes an ai-toolkit job for arch ``minimax_h3_ref2va`` (SDR control videos as the <Video 1> reference, log-HDR
targets, loss on the target tokens only, LoRA on qkv_proj / out_proj / fc1 / fc2) and runs ``<python> run.py <yaml>``
in the ai-toolkit folder as a background process; the HDR jobs panel shows its log and loss curve and can cancel it.
Same job name + output folder = continue from the last checkpoint. The work is in mmx_utils/hdr_a29_train.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import hdr_a29_jobs as J  # noqa: E402
from mmx_utils import hdr_a29_train as T  # noqa: E402

CATEGORY = "MiniMax H3/HDR"
J.register_routes()


def _resolve(path: str) -> str:
    p = (path or "").strip().strip('"')
    if not p:
        raise ValueError("a folder is required")
    if os.path.isabs(p):
        return os.path.normpath(p)
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:  # noqa: BLE001
        base = os.getcwd()
    return os.path.normpath(os.path.join(base, p))


class MiniMaxH3_HDRTrainICLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_HDRTrainICLoRA",
            display_name="H3 HDR · Train IC-LoRA",
            category=CATEGORY,
            description=("Trains the HDR in-context LoRA with ai-toolkit (ostris, MIT; run as a separate program): "
                         "SDR clip = <Video 1> reference, log-HDR clip = target, loss on the target only, LoRA on "
                         "qkv_proj / out_proj / fc1 / fc2. Checks first that your ai-toolkit has MiniMax H3 support "
                         "(else: run `git pull` in that folder). Runs in the background with a live log + loss curve; "
                         "Cancel stops it; the same job name continues from its last checkpoint."),
            search_aliases=["hdr lora training", "ai-toolkit", "train ic lora", "minimax h3 train"],
            inputs=[
                io.String.Input("dataset_folder", default="hdr_dataset/pairs",
                                tooltip="Make Pairs output (targets/ + control/). Relative = inside ComfyUI's output."),
                io.String.Input("aitoolkit_dir", default="",
                                tooltip="Your ai-toolkit folder (git clone of ostris/ai-toolkit, with MiniMax H3)."),
                io.String.Input("python", default="",
                                tooltip="Python of the ai-toolkit environment; empty = its venv / .venv, else "
                                        "ComfyUI's Python."),
                io.String.Input("job_name", default="h3_hdr_iclora_v1",
                                tooltip="Names the run and its checkpoints. Same name = resume."),
                io.String.Input("output_folder", default="hdr_lora",
                                tooltip="ai-toolkit training_folder. Relative = inside ComfyUI's output."),
                io.Int.Input("steps", default=3000, min=1, max=1_000_000,
                             tooltip="Total optimiser steps written to the YAML (train.steps; batch size 1, no "
                                     "gradient accumulation)."),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=1e-2, step=1e-5, round=False,
                               tooltip="AdamW8bit learning rate written to the YAML (train.lr)."),
                io.Int.Input("save_every", default=250, min=1, max=100_000,
                             tooltip="Write a LoRA checkpoint (bf16) every N steps; ai-toolkit keeps the newest 4. "
                                     "Resume continues from the newest one."),
                io.Int.Input("rank", default=16, min=1, max=512, tooltip="ref2va default 16."),
                io.Int.Input("alpha", default=16, min=1, max=512, tooltip="ref2va default 16."),
                io.Int.Input("resolution", default=768, min=256, max=2048, step=32,
                             tooltip="ai-toolkit caps each clip's area at resolution x resolution (aspect kept): "
                                     "768 trains 1344x768 clips at about 1024x576; 1024 trains them full size."),
                io.Combo.Input("partition", options=list(T.PARTITIONS), default="ref2va_pruned",
                               tooltip="Which ref2va checkpoint of Comfy-Org/MiniMax-H3 to train on."),
                io.Boolean.Input("resume", default=True,
                                 tooltip="On: continue from the newest checkpoint of this job name. Off: refuse when "
                                         "one exists (pick a new name to start fresh)."),
                io.Boolean.Input("dry_run", default=False,
                                 tooltip="Check everything and write the YAML, but do not start training."),
                io.Boolean.Input("wait", default=False,
                                 tooltip="Off (default): training runs in the background and the queue moves on. "
                                         "On: the node waits until training ends (ComfyUI's Cancel stops it)."),
            ],
            outputs=[io.String.Output("report", tooltip="Text: ai-toolkit commit and H3 capability checks, dataset "
                                                        "pairs / frames / size, resume point, YAML path, then the "
                                                        "background job id and log (and final steps / loss when wait "
                                                        "is on)."),
                     io.String.Output("yaml_path", tooltip="Path of the ai-toolkit job YAML written under "
                                                           "<output_folder>/a29_runs/<job_name>/ (also in a dry run).")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kw):
        out = kw.get("output_folder", "")
        try:
            sig = J.folder_signature(os.path.join(_resolve(out), kw.get("job_name", "")))
        except Exception:  # noqa: BLE001
            sig = ""
        return "|".join(f"{k}={kw[k]}" for k in sorted(kw)) + "|" + sig

    @classmethod
    def execute(cls, dataset_folder, aitoolkit_dir, python, job_name, output_folder, steps, learning_rate, save_every,
                rank, alpha, resolution, partition, resume, dry_run, wait) -> io.NodeOutput:
        prep = T.prepare(dataset_folder=_resolve(dataset_folder), toolkit_dir=(aitoolkit_dir or "").strip().strip('"'),
                         output_dir=_resolve(output_folder), name=job_name.strip(), resume=resume, steps=steps,
                         lr=learning_rate, save_every=save_every, rank=rank, alpha=alpha, resolution=resolution,
                         partition=partition)
        report = T.verified_text(prep)
        if dry_run:
            report += "\n\ndry run: YAML written, training NOT started."
            return io.NodeOutput(report, prep["yaml"], ui={"text": [report]})
        job = T.launch(prep, (python or "").strip().strip('"') or None)
        report += f"\n\ntraining started in the background: job {job.id}, log {prep['log']}"
        if wait:
            progress, interrupted = J.comfy_waiter()
            try:
                J.wait_for(job, progress=progress, interrupted=interrupted, poll=1.0)
            except J.JobCancelled:
                import comfy.model_management as mm
                mm.throw_exception_if_processing_interrupted()
                raise
            if job.status != "done":
                raise RuntimeError(f"training ended with status {job.status}: {job.error}\n" + "\n".join(job.tail(20)))
            last = job.loss[-1] if job.loss else None
            report += f"\nfinished: {job.done}/{job.total} steps" + (f", last loss {last[1]:.4g}" if last else "")
            ck = T.latest_checkpoint(_resolve(output_folder), job_name.strip())
            report += f"\nlast checkpoint: {ck}"
        return io.NodeOutput(report, prep["yaml"], ui={**J.ui_payload(job), "text": [report]})


NODES = [MiniMaxH3_HDRTrainICLoRA]
