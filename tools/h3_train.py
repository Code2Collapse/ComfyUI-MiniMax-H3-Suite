"""Reproducible H3 HDR LoRA training through a PINNED ai-toolkit (ledger L9.209, A26 3.4).

The trainer is ostris/ai-toolkit (MIT), used as an external program at one exact commit. Nothing of it is copied
here. Configs: docs/training/h3_hdr/goal_a_log_lora.yaml (Goal A: generate in a log code) and
goal_b_sdr2hdr_iclora.yaml (Goal B: SDR-conditioned reconstruction as a side-by-side in-context LoRA).
Documentation: docs/training/h3_hdr/README.md.

    python tools/h3_train.py check     --toolkit /opt/ai-toolkit --config docs/training/h3_hdr/goal_a_log_lora.yaml
    python tools/h3_train.py val-items --dataset /data/h3_hdr --out /data/h3_hdr/val_items   # validation images
    python tools/h3_train.py launch    --toolkit /opt/ai-toolkit --config my_goal_a.yaml --seed 1234 --run runs/a1
    python tools/h3_train.py export-check runs/a1/output/h3_hdr_log_acescct_v1/h3_hdr_log_acescct_v1.safetensors

``check`` refuses a toolkit at another commit (``--allow-unpinned`` records the difference instead), keys the pinned
code does not read, frame counts off the 17k+5 grid, sides that are not multiples of 32, LoRA targets that would
touch audio-only or modality-mixing modules, and a dataset folder that is not a TRAIN split of an h3_hdr_dataset v2
build. ``launch`` writes run_manifest.json (pin, config + dataset hashes, model file identities, Python / torch /
CUDA, pip freeze, seed, resume point) before starting the trainer, and tees its log. ``export-check`` validates the
saved LoRA for the H3 nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

AITK_REPO = "https://github.com/ostris/ai-toolkit"
AITK_PIN = "4a7f28532ed6fe81d9f8c00718878bb3b8739015"          # main, 2026-11-10 (minimax_h3 + H3 Image adapter)
AITK_PIN_DATE = "2026-11-10"

H3_BLOCKS = 50                                                 # MiniMaxH3TransformerParams.num_layers at the pin
# the per-block linears shared by text, video and audio rows (one joint sequence; modality only picks AdaLN rows)
ADAPTED = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
# never adapted: audio-only modules, the per-modality AdaLN table, embeddings / refiner / output heads
REQUIRED_IGNORES = ("audio", "adaln_proj", "token_refiner", "time_embedder", "condition_proj", "patch_proj",
                    "final_layer")
PARTITIONS = ("fl2va", "fl2va_pruned", "ref2va", "ref2va_pruned")
LOSS_TYPES = ("mse", "mae", "wavelet", "band", "pixelspace", "mean_flow", "pseudo_huber")   # TrainConfig at the pin
GOALS = ("A", "B")


class CheckError(SystemExit):
    pass


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────
def load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("Reading the config needs PyYAML (pip install pyyaml).") from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def toolkit_commit(toolkit: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", toolkit, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def process_of(cfg: dict) -> dict:
    try:
        return cfg["config"]["process"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise CheckError("The config has no config.process[0] (ai-toolkit job layout).") from exc


def goal_of(cfg: dict) -> str:
    g = str(((cfg.get("meta") or {}).get("c2c_goal")) or "")
    if g not in GOALS:
        raise CheckError("Set meta.c2c_goal to 'A' (log-code generation) or 'B' (SDR-conditioned reconstruction).")
    return g


def bucket(width: int, height: int, resolution: int, divisibility: int = 32) -> tuple[int, int]:
    """The training size ai-toolkit gives a clip: area capped at resolution^2, aspect kept, never upscaled, sides
    rounded to the divisibility (H3: 32). Same arithmetic as toolkit/buckets.py get_bucket_for_image_size at the pin
    (ai-toolkit, MIT)."""
    import math
    total, cap = width * height, resolution * resolution
    target = min(total, cap)
    s = (target / total) ** 0.5
    wr, hr = width * s / divisibility, height * s / divisibility
    cands = [(math.floor(wr) * divisibility, math.floor(hr) * divisibility),
             (math.floor(wr) * divisibility, math.ceil(hr) * divisibility),
             (math.ceil(wr) * divisibility, math.floor(hr) * divisibility),
             (math.ceil(wr) * divisibility, math.ceil(hr) * divisibility)]
    ok = [(w, h) for w, h in cands if w > 0 and h > 0 and w * h <= cap] or         [(max(divisibility, math.floor(wr) * divisibility), max(divisibility, math.floor(hr) * divisibility))]
    return min(ok, key=lambda wh: abs(wh[0] * wh[1] - target))


# ── check ───────────────────────────────────────────────────────────────────────────────────────────────────────
def check(cfg: dict, *, commit: str | None, allow_unpinned: bool = False, dataset_root: str | None = None) -> dict:
    """Validate a config against the pinned trainer + the H3 HDR rules. Returns a report; raises CheckError."""
    problems, notes = [], []
    if commit != AITK_PIN:
        msg = f"ai-toolkit is at {commit or 'an unknown commit'}, the integration is pinned to {AITK_PIN}"
        (notes if allow_unpinned else problems).append(msg)
    if cfg.get("job") != "extension":
        problems.append("job must be 'extension' (ai-toolkit's trainer job)")
    goal = goal_of(cfg)
    p = process_of(cfg)
    if p.get("type") != "sd_trainer":
        problems.append("process type must be 'sd_trainer'")
    model = p.get("model") or {}
    if model.get("arch") != "minimax_h3":
        problems.append("model.arch must be 'minimax_h3' (t2v / first-frame i2v) for both goals")
    part = (model.get("model_kwargs") or {}).get("partition")
    if part not in PARTITIONS:
        problems.append(f"model.model_kwargs.partition must be one of {PARTITIONS} (the code's default is fl2va_pruned)")
    if (model.get("model_kwargs") or {}).get("sample_audio"):
        problems.append("sample_audio must be false: the HDR adapters are video-only")
    net = p.get("network") or {}
    if net.get("type") != "lora":
        problems.append("network.type must be 'lora'")
    rank, alpha = net.get("linear"), net.get("linear_alpha")
    if not isinstance(rank, int) or rank < 1 or not isinstance(alpha, (int, float)):
        problems.append("network.linear (rank) and network.linear_alpha must be set")
    if net.get("transformer_only", True) is not True:
        problems.append("network.transformer_only must stay true (adapt the 50 DiT blocks only)")
    nk = net.get("network_kwargs") or {}
    ign = [str(x) for x in (nk.get("ignore_if_contains") or [])]
    missing = [r for r in REQUIRED_IGNORES if r not in ign]
    if missing:
        problems.append(f"network.network_kwargs.ignore_if_contains must list {missing}: those modules carry audio or "
                        "mix modalities")
    for ds in p.get("datasets") or []:
        if ds.get("do_audio"):
            problems.append("datasets[].do_audio must be false: the HDR target is video-only")
        n = ds.get("num_frames")
        if not isinstance(n, int) or n < 22 or (n - 5) % 17:
            problems.append(f"datasets[].num_frames {n} is not on H3's 17k+5 grid (22, 39, 56 ...)")
        res = ds.get("resolution")
        res = res if isinstance(res, list) else [res]
        if any(not isinstance(r, int) or r < 256 for r in res):
            problems.append(f"datasets[].resolution {res}: integers >= 256 (each caps a clip's area at R x R)")
            res = []
        folder = str(ds.get("folder_path", ""))
        tail = Path(folder.replace("\\", "/"))
        want = ("trackc", "train") if goal == "B" else ("train",)
        if goal == "A" and tail.name != "train":
            problems.append(f"Goal A trains on the dataset's train/ folder, not {folder}")
        if goal == "B" and not (tail.name == "train" and tail.parent.name == "trackc"):
            problems.append(f"Goal B trains on trackc/train/ (side-by-side SDR | log pairs), not {folder}")
        if dataset_root:
            m = Path(dataset_root) / "manifest.json"
            if not m.is_file():
                problems.append(f"{dataset_root} has no manifest.json (build it with tools/h3_hdr_dataset.py)")
            else:
                man = json.loads(m.read_text(encoding="utf-8"))
                if man.get("version", 1) < 2 or not (man.get("integrity") or {}).get("ok"):
                    problems.append("the dataset manifest is older than v2 or its integrity check failed")
                if man.get("frames") != n:
                    problems.append(f"num_frames {n} differs from the dataset's {man.get('frames')}")
                if goal == "B" and not any(s.get("trackc") for c in man.get("clips", []) for s in c.get("sdr", [])):
                    problems.append("Goal B needs a dataset built with --trackc")
                notes.append(f"dataset: {want}, {len(man.get('clips', []))} clips, code {man.get('code')}")
                w, h = man.get("size", [0, 0])
                if goal == "B":
                    w *= 2                                                # [SDR | log] side by side
                if w and h:
                    for r in res:
                        notes.append(f"resolution {r}: {w}x{h} clips train at {'x'.join(map(str, bucket(w, h, r)))}")
    tr = p.get("train") or {}
    lt = tr.get("loss_type", "mse")
    if lt not in LOSS_TYPES:
        problems.append(f"train.loss_type {lt!r} does not exist at the pinned commit ({', '.join(LOSS_TYPES)})")
    for k in ("temporal_loss", "temporal_loss_weight", "hdr_loss", "perceptual_loss"):
        if k in tr:
            problems.append(f"train.{k} is not a setting of the pinned trainer: it would be silently ignored")
    if (p.get("save") or {}).get("save_every") is None:
        problems.append("save.save_every must be set (checkpoint interval)")
    rep = {"ok": not problems, "goal": goal, "problems": problems, "notes": notes, "pin": AITK_PIN,
           "toolkit_commit": commit, "loss_type": lt, "rank": rank, "alpha": alpha, "partition": part,
           "adapted_modules": [f"blocks.N.{m}" for m in ADAPTED], "expected_lora_modules": H3_BLOCKS * len(ADAPTED)}
    if problems:
        raise CheckError("config check failed:\n  - " + "\n  - ".join(problems))
    return rep


# ── export check ────────────────────────────────────────────────────────────────────────────────────────────────
_KEY = re.compile(r"^(?:diffusion_model\.)?blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|attn\.to_gate_compress|"
                  r"mlp\.fc1|mlp\.fc2)\.(lora_A|lora_B|lora_down|lora_up|alpha)(?:\.weight)?$")


def export_check(path: str) -> dict:
    """A saved LoRA for the H3 nodes: ComfyUI key layout, only video-path block linears, finite factors, one rank."""
    from safetensors import safe_open  # type: ignore[import-not-found]
    import numpy as np
    problems, modules, ranks, prefixed, n_keys = [], {}, set(), 0, 0
    with safe_open(path, framework="np") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            m = _KEY.match(k)
            if not m:
                problems.append(f"unexpected key {k} (not a video-path block linear)")
                continue
            n_keys += 1
            prefixed += k.startswith("diffusion_model.")
            t = f.get_tensor(k)
            if not np.isfinite(t.astype(np.float32)).all():
                problems.append(f"{k} has non-finite values")
            mod = f"blocks.{m.group(1)}.{m.group(2)}"
            modules.setdefault(mod, set()).add(m.group(3))
            if m.group(3) in ("lora_A", "lora_down"):
                ranks.add(int(t.shape[0]))
    for mod, parts in modules.items():
        if not ({"lora_A", "lora_B"} <= parts or {"lora_down", "lora_up"} <= parts):
            problems.append(f"{mod} lacks one of its two factors")
    if len(ranks) > 1:
        problems.append(f"mixed ranks {sorted(ranks)}")
    if 0 < prefixed < n_keys:
        problems.append("some keys use the diffusion_model. prefix and some do not")
    blocks = sorted({int(m.split(".")[1]) for m in modules})
    rep = {"ok": not problems, "file": os.path.basename(path), "sha256": sha256_file(path), "modules": len(modules),
           "expected_modules": H3_BLOCKS * len(ADAPTED), "coverage": len(modules) / (H3_BLOCKS * len(ADAPTED)),
           "blocks": [blocks[0], blocks[-1]] if blocks else None, "rank": sorted(ranks)[0] if ranks else None,
           "comfy_prefix": bool(prefixed), "metadata_keys": sorted(meta)[:20], "problems": problems[:20]}
    return rep


# ── validation items ────────────────────────────────────────────────────────────────────────────────────────────
def val_items(dataset: str, out: str, limit: int = 8) -> list[dict]:
    """First frame of each val clip (log-code PNG, exactly the training codes) + its caption, as ai-toolkit
    validation_config.validation_items (image-based loss at fixed sigmas: a cheap tracker, not the video QC)."""
    import av  # type: ignore[import-not-found]
    from PIL import Image
    man = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    os.makedirs(out, exist_ok=True)
    items = []
    for c in [c for c in man["clips"] if c["split"] == "val"][:limit]:
        src = Path(dataset) / c["target"]["file"]
        with av.open(str(src)) as cont:
            fr = next(cont.decode(video=0)).to_ndarray(format="rgb24")
        p = Path(out) / f"{c['id']}_f0.png"
        Image.fromarray(fr).save(p)
        items.append({"image_path": str(p), "prompt": c["caption"]})
    return items


# ── launch ──────────────────────────────────────────────────────────────────────────────────────────────────────
def latest_checkpoint(output_dir: str, name: str) -> str | None:
    d = Path(output_dir) / name
    cands = sorted(d.glob(f"{name}*.safetensors"), key=lambda p: p.stat().st_mtime) if d.is_dir() else []
    return str(cands[-1]) if cands else None


def model_identities(p: dict, full: bool) -> dict:
    out = {}
    for k, v in ((p.get("model") or {}).get("model_kwargs") or {}).items():
        if k.endswith("_path") and isinstance(v, str) and os.path.isfile(v):
            out[k] = {"file": os.path.basename(v), "bytes": os.path.getsize(v),
                      ("sha256" if full else "sha256_first_mib"): sha256_file(v, None if full else 1 << 20)}
    return out


def launch(a) -> int:
    cfg = load_yaml(a.config)
    commit = toolkit_commit(a.toolkit)
    rep = check(cfg, commit=commit, allow_unpinned=a.allow_unpinned, dataset_root=a.dataset)
    p = process_of(cfg)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    out_dir = str(run / "output")
    p["training_folder"] = out_dir                                    # every run writes into its own folder
    eff = run / "config.yaml"
    import yaml  # type: ignore[import-not-found]
    eff.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    freeze = subprocess.run([a.python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    ver = subprocess.run([a.python, "-c", "import torch, sys; print(sys.version.split()[0], torch.__version__, "
                          "torch.version.cuda, torch.cuda.is_available())"], capture_output=True, text=True).stdout
    name = cfg["config"]["name"]
    manifest = {"tool": "h3_train", "version": 1, "goal": rep["goal"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "trainer": {"repo": AITK_REPO, "pin": AITK_PIN, "pin_date": AITK_PIN_DATE, "commit": commit},
                "config_sha256": sha256_file(str(eff)), "config": str(eff), "seed": a.seed,
                "seeded": "python random / numpy / torch (+ cuda) before the trainer starts; ai-toolkit at the pin has "
                          "no training seed setting, and CUDA kernels can still be non-deterministic",
                "dataset_manifest_sha256": sha256_file(str(Path(a.dataset) / "manifest.json")) if a.dataset else None,
                "models": model_identities(p, a.full_hash), "python_torch_cuda": ver.strip(),
                "platform": platform.platform(), "resume_from": latest_checkpoint(out_dir, name),
                "check": rep}
    (run / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if a.dry_run:
        print(json.dumps({"run_manifest": str(run / "run_manifest.json")}))
        return 0
    boot = ("import os, random, sys, runpy\n"
            f"seed = {int(a.seed)}\n"
            "random.seed(seed)\n"
            "import numpy; numpy.random.seed(seed)\n"
            "import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)\n"
            f"sys.argv = ['run.py', {str(eff)!r}]\n"
            f"os.chdir({a.toolkit!r}); sys.path.insert(0, {a.toolkit!r})\n"
            "runpy.run_path('run.py', run_name='__main__')\n")
    env = dict(os.environ, PYTHONHASHSEED=str(int(a.seed)))
    with open(run / "train.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen([a.python, "-u", "-c", boot], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    manifest["finished"], manifest["exit_code"] = time.strftime("%Y-%m-%dT%H:%M:%S"), code
    manifest["last_checkpoint"] = latest_checkpoint(out_dir, name)
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--toolkit", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--dataset", help="the h3_hdr_dataset output folder (with manifest.json)")
    c.add_argument("--allow-unpinned", action="store_true")
    v = sub.add_parser("val-items")
    v.add_argument("--dataset", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--limit", type=int, default=8)
    e = sub.add_parser("export-check")
    e.add_argument("lora")
    ln = sub.add_parser("launch")
    ln.add_argument("--toolkit", required=True)
    ln.add_argument("--config", required=True)
    ln.add_argument("--dataset")
    ln.add_argument("--run", required=True, help="a new folder for this run (manifest, log, checkpoints)")
    ln.add_argument("--seed", type=int, required=True)
    ln.add_argument("--python", default=sys.executable, help="the Python of the ai-toolkit environment")
    ln.add_argument("--full-hash", action="store_true")
    ln.add_argument("--allow-unpinned", action="store_true")
    ln.add_argument("--dry-run", action="store_true", help="write the manifest, do not start training")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        rep = check(load_yaml(a.config), commit=toolkit_commit(a.toolkit), allow_unpinned=a.allow_unpinned,
                    dataset_root=a.dataset)
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "val-items":
        print(json.dumps({"validation_items": val_items(a.dataset, a.out, a.limit)}, indent=1))
        return 0
    if a.cmd == "export-check":
        rep = export_check(a.lora)
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1
    return launch(a)


if __name__ == "__main__":
    sys.exit(main())
