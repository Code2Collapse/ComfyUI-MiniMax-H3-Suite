"""A29 HDR IC-LoRA training through ai-toolkit (owner order A29 node 16, design docs/hdr/A29_DESIGN.md section 1).

ai-toolkit (ostris, MIT) is used as an external program, never imported or copied. What its pinned commit
(tools/h3_train.py AITK_PIN = 4a7f285, read with ``git show``) provides, and this module relies on:

- arch ``minimax_h3_ref2va`` (MinimaxH3Ref2VAModel, ``supports_video_control_images = True``): a dataset's
  ``control_path`` holding VIDEOS with the target's file name is encoded as reference rows ("<Video 1>");
- the reference rows are prepended as conditioning and sliced off before the loss (``video_pred[:, num_cond:]``),
  so the loss is on the target tokens only: no trainer change needed;
- ``network.network_kwargs.only_if_contains`` / ``ignore_if_contains`` go to LoRASpecialNetwork unchanged;
- progress: a tqdm bar whose postfix is ``lr: 1.0e-04 loss: 1.234e-01`` (BaseSDTrainProcess + SDTrainer loss_dict);
- resume: on start, the newest ``<training_folder>/<name>/<name>*.safetensors`` is loaded and its step restored
  from the file's metadata (BaseSDTrainProcess.get_latest_save_path / load_training_state_from_metadata). Same
  job name + same output folder = continue; a new name starts fresh.

The pin check, manifest helpers and checkpoint lookup are reused from tools/h3_train.py (L9.209).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from . import hdr_a29_jobs as J
except ImportError:
    from mmx_utils import hdr_a29_jobs as J

_PKG = Path(__file__).resolve().parents[1]
ARCH = "minimax_h3_ref2va"
PARTITIONS = ("ref2va_pruned", "ref2va")                         # MinimaxH3Ref2VAModel accepts only these
LAYER_MAP = ["qkv_proj", "out_proj", "fc1", "fc2"]              # same as the Open Regen IC LoRA
IGNORE = ["adaln_proj"]                                         # ref2va UI default at the pin
ASSISTANT_LORA = "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v2.safetensors"
BASE_MODEL = "Comfy-Org/MiniMax-H3"
MODEL_FILE = Path("extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py")
# what the trainer's own source must contain for the HDR IC-LoRA to work (checked in the owner's clone)
CAPABILITIES = {
    "ref2va arch": "class MinimaxH3Ref2VAModel",
    "reference videos from control_path": "supports_video_control_images = True",
    "loss on target tokens only": "video_pred = video_pred[:, num_cond:]",
}
_TQDM = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?\blr:\s*([0-9.eE+-]+)\s+loss:\s*([0-9.eE+-]+|nan|inf)", re.I)
_STEP = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


def h3_train_tool():
    """tools/h3_train.py as a module (pin, toolkit_commit, latest_checkpoint, bucket, sha256_file)."""
    name = "_mmx_tools_h3_train"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _PKG / "tools" / "h3_train.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


class ToolkitNotReady(RuntimeError):
    """The ai-toolkit folder cannot run the H3 HDR IC-LoRA (missing, or a commit without minimax_h3)."""


def check_toolkit(toolkit_dir: str, commit: str | None = "auto") -> dict:
    """Refuse a folder that is not an ai-toolkit with minimax_h3; report the commit vs the pin and which of the
    needed capabilities its source shows. ``commit='auto'`` asks git."""
    T = h3_train_tool()
    d = Path(toolkit_dir or "")
    if not toolkit_dir or not d.is_dir():
        raise ToolkitNotReady(f"ai-toolkit folder not found: {toolkit_dir!r}. Clone https://github.com/ostris/ai-toolkit "
                              "and put its folder in aitoolkit_dir.")
    if not (d / "run.py").is_file():
        raise ToolkitNotReady(f"{d} has no run.py - that is not an ai-toolkit checkout.")
    src = d / MODEL_FILE
    if not src.is_file():
        raise ToolkitNotReady(
            f"{d} has no {MODEL_FILE.parent.as_posix()} (MiniMax H3 support). Your ai-toolkit is older than the commit "
            f"that added it. Update it: open a terminal in {d} and run `git pull` (H3 support is in "
            f"{T.AITK_PIN[:7]} and later), then run this node again.")
    text = src.read_text(encoding="utf-8", errors="replace")
    caps = {k: (v in text) for k, v in CAPABILITIES.items()}
    missing = [k for k, ok in caps.items() if not ok]
    if missing:
        raise ToolkitNotReady(f"{src} lacks: {', '.join(missing)}. Update ai-toolkit (`git pull` in {d}).")
    if commit == "auto":
        commit = T.toolkit_commit(str(d))
    return {"toolkit": str(d), "commit": commit, "pin": T.AITK_PIN, "at_pin": commit == T.AITK_PIN,
            "capabilities": caps}


def default_python(toolkit_dir: str) -> str:
    """The ai-toolkit venv's Python when there is one (venv / .venv), else this Python."""
    for rel in ("venv/Scripts/python.exe", ".venv/Scripts/python.exe", "venv/bin/python", ".venv/bin/python"):
        p = Path(toolkit_dir) / rel
        if p.is_file():
            return str(p)
    return sys.executable


def dataset_info(dataset_folder: str) -> dict:
    """targets/ + control/ of a Make Pairs dataset: pair count, frame counts, size; refuses mismatched names."""
    d = Path(dataset_folder)
    tdir, cdir = d / "targets", d / "control"
    if not tdir.is_dir() or not cdir.is_dir():
        raise FileNotFoundError(f"{d} has no targets/ and control/ folders - run HDR Dataset: Make Pairs first")
    t = sorted(p.stem for p in tdir.glob("*.mp4"))
    c = sorted(p.stem for p in cdir.glob("*.mp4"))
    if not t:
        raise FileNotFoundError(f"{tdir} holds no .mp4 targets")
    if t != c:
        raise ValueError(f"targets/ and control/ must hold the SAME file names (ai-toolkit pairs them by name); "
                         f"only in targets: {sorted(set(t) - set(c))[:5]}, only in control: {sorted(set(c) - set(t))[:5]}")
    no_caption = [s for s in t if not (tdir / f"{s}.txt").is_file()]
    frames, size = set(), None
    meta = d / "meta.jsonl"
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            frames.add(int(row.get("frames", 0)))
            size = size or row.get("size")
    return {"pairs": len(t), "frames": sorted(frames), "size": size, "missing_captions": no_caption,
            "targets": str(tdir), "control": str(cdir)}


def build_config(*, name: str, dataset_folder: str, output_dir: str, steps: int = 3000, lr: float = 1e-4,
                 save_every: int = 250, rank: int = 16, alpha: int = 16, resolution: int = 768,
                 num_frames: int = 73, partition: str = "ref2va_pruned", assistant_lora: str = ASSISTANT_LORA,
                 caption_dropout: float = 0.0, max_saves: int = 4) -> dict:
    """The ai-toolkit job (config.process[0]) for the HDR IC-LoRA: ref2va, SDR control videos as <Video 1>,
    log-HDR targets, LoRA on qkv_proj / out_proj / fc1 / fc2 (rank / alpha 16, ref2va UI defaults)."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    if num_frames < 22 or (num_frames - 5) % 17:
        raise ValueError(f"num_frames {num_frames} is not on the 17k+5 grid")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
        raise ValueError("job_name may use letters, digits, '_', '-' and '.' only (it names the checkpoint files)")
    d = Path(dataset_folder)
    return {
        "job": "extension",
        "meta": {"name": name, "version": "1.0", "c2c": "A29 HDR IC-LoRA (MiniMax H3 Suite)",
                 "c2c_encoding": "diffhdr_log_16384"},
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(output_dir),
                "device": "cuda",
                "trigger_word": None,
                "performance_log_every": 10,
                "network": {
                    "type": "lora", "linear": int(rank), "linear_alpha": int(alpha), "transformer_only": True,
                    "network_kwargs": {"only_if_contains": list(LAYER_MAP), "ignore_if_contains": list(IGNORE)},
                },
                "save": {"dtype": "bf16", "save_every": int(save_every), "max_step_saves_to_keep": int(max_saves),
                         "push_to_hub": False},
                "datasets": [{
                    "folder_path": str(d / "targets"),             # log-HDR targets (+ captions)
                    "control_path": str(d / "control"),            # SDR reference videos, same file names
                    "caption_ext": "txt",
                    "caption_dropout_rate": float(caption_dropout),
                    "cache_latents_to_disk": True,
                    "resolution": [int(resolution)],
                    "num_frames": int(num_frames),
                    "auto_frame_count": True,
                    "fps": 24,
                    "do_audio": False,
                    "shrink_video_to_frames": True,
                }],
                "train": {
                    "batch_size": 1, "steps": int(steps), "gradient_accumulation": 1, "train_unet": True,
                    "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                    "timestep_type": "shift", "optimizer": "adamw8bit", "lr": float(lr),
                    "optimizer_params": {"weight_decay": 1e-4}, "loss_type": "mse", "dtype": "bf16",
                    "cache_text_embeddings": True, "disable_sampling": True,
                },
                "model": {
                    "name_or_path": BASE_MODEL, "arch": ARCH, "quantize": True, "qtype": "convrot8",
                    "quantize_te": True, "qtype_te": "nvfp4", "low_vram": True,
                    "assistant_lora_path": assistant_lora or None,
                    "model_kwargs": {"partition": partition},
                },
                "logging": {"log_every": 1},
            }],
        },
    }


def write_yaml(cfg: dict, path: str | os.PathLike) -> str:
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(p)


def parse_line(line: str) -> dict | None:
    """One trainer log line -> {step, total, lr?, loss?} (tqdm bar, e.g.
    ``h3_hdr: 12%|#  | 120/1000 [01:23<10:00, 1.43s/it, lr: 1.0e-04 loss: 1.234e-01]``), else None."""
    m = _TQDM.search(line)
    if m:
        return {"step": int(m.group(1)), "total": int(m.group(2)), "lr": float(m.group(3)),
                "loss": float(m.group(4))}
    m = _STEP.search(line)
    if m:
        return {"step": int(m.group(1)), "total": int(m.group(2))}
    return None


def parse_into_job(job: "J.Job", line: str) -> None:
    r = parse_line(line)
    if not r:
        return
    job.set_progress(r["step"], r["total"], f"step {r['step']}/{r['total']}"
                     + (f", loss {r['loss']:.4g}" if "loss" in r else ""))
    if "loss" in r:
        job.add_loss(r["step"], r["loss"])


def parse_log(text: str) -> list[tuple[int, float]]:
    """(step, loss) pairs of a whole log (tqdm \\r redraws included); the last value per step wins."""
    out: dict[int, float] = {}
    for line in re.split(r"[\r\n]+", text):
        r = parse_line(line)
        if r and "loss" in r:
            out[r["step"]] = r["loss"]
    return sorted(out.items())


def latest_checkpoint(output_dir: str, name: str) -> str | None:
    return h3_train_tool().latest_checkpoint(output_dir, name)


def prepare(*, dataset_folder: str, toolkit_dir: str, output_dir: str, name: str, resume: bool = True,
            commit: str | None = "auto", **cfg_kw) -> dict:
    """Check everything, write the YAML + a run manifest; return what launch() needs. Raises in plain words."""
    tk = check_toolkit(toolkit_dir, commit=commit)
    ds = dataset_info(dataset_folder)
    if len(ds["frames"]) > 1:
        raise ValueError(f"the dataset mixes frame counts {ds['frames']}: build it with one HDRI -> Clips frame count")
    frames = ds["frames"][0] if ds["frames"] else int(cfg_kw.pop("num_frames", 73))
    cfg_kw.pop("num_frames", None)
    out = Path(output_dir)
    ckpt = latest_checkpoint(str(out), name)
    if ckpt and not resume:
        raise FileExistsError(f"{ckpt} exists: ai-toolkit would CONTINUE from it. Turn resume on to continue, or pick "
                              "a new job_name to start fresh.")
    cfg = build_config(name=name, dataset_folder=dataset_folder, output_dir=str(out), num_frames=frames, **cfg_kw)
    # NOT inside <output>/<name>/: ai-toolkit's resume globs "<name>*" there, and "<name>.yaml" would match
    run = out / "a29_runs" / name
    yaml_path = write_yaml(cfg, run / f"{name}.yaml")
    T = h3_train_tool()
    w, h = (ds["size"] or [0, 0])
    bucket = T.bucket(w, h, cfg["config"]["process"][0]["datasets"][0]["resolution"][0]) if w and h else None
    manifest = {"tool": "hdr_a29_train", "version": 1, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "toolkit": tk, "dataset": ds, "yaml": yaml_path, "config_sha256": T.sha256_file(yaml_path),
                "resume_from": ckpt, "train_size": bucket}
    (run / "a29_run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {"yaml": yaml_path, "run_dir": str(run), "log": str(run / "train.log"), "resume_from": ckpt,
            "toolkit": tk, "dataset": ds, "frames": frames, "train_size": bucket, "config": cfg}


def launch(prep: dict, python: str | None = None) -> "J.Job":
    """Start ``<python> run.py <yaml>`` in the ai-toolkit folder as a background job (log + loss parsed live)."""
    tk = prep["toolkit"]["toolkit"]
    py = python or default_python(tk)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    name = prep["config"]["config"]["name"]
    return J.start_process("train", f"HDR IC-LoRA {name}", [py, "run.py", prep["yaml"]], cwd=tk, env=env,
                           log_path=prep["log"], parse=parse_into_job, key=f"train:{prep['run_dir']}")


def verified_text(prep: dict) -> str:
    tk = prep["toolkit"]
    lines = [f"ai-toolkit: {tk['toolkit']} at {str(tk['commit'] or 'unknown')[:7]} "
             f"({'= the pinned commit' if tk['at_pin'] else 'NOT the pinned ' + tk['pin'][:7] + ': untested there'})"]
    for k, ok in tk["capabilities"].items():
        lines.append(f"  {'OK ' if ok else 'NO '} {k} (found in {MODEL_FILE.as_posix()})")
    ds = prep["dataset"]
    lines.append(f"dataset: {ds['pairs']} pairs, frames {ds['frames'] or '?'}, size {ds['size']}, "
                 f"captions missing: {len(ds['missing_captions'])}")
    if prep.get("train_size"):
        lines.append(f"clips train at {prep['train_size'][0]}x{prep['train_size'][1]} (ai-toolkit bucket)")
    lines.append(f"resume: {'continues from ' + prep['resume_from'] if prep['resume_from'] else 'fresh start'}")
    lines.append(f"YAML: {prep['yaml']}")
    lines.append("NOT verified here: a real training run (needs the H3 weights + a CUDA GPU).")
    return "\n".join(lines)
