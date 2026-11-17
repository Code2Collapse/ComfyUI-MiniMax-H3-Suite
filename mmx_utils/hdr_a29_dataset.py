"""A29 HDR training-set builder for the H3 HDR IC-LoRA (owner order A29 nodes 12-15, design docs/hdr/A29_DESIGN.md).

Separate from ``mmx_utils/hdr_dataset.py`` (L9.180 / L9.208, the ACEScg / side-by-side Track C builder used by
tools/h3_hdr_dataset.py), which stays untouched. This module builds the A29 layout ai-toolkit's
``minimax_h3_ref2va`` arch reads as an in-context pair:

    <dataset>/targets/<clip>.mp4   log-HDR target   (hdr_math 'diffhdr_log_16384', lossless RGB, libx264rgb crf 0)
    <dataset>/targets/<clip>.txt   caption          (HDR Prompt template: scene / lighting / camera)
    <dataset>/control/<clip>.mp4   SDR reference    (sRGB, 8-bit h264, crf 16..28)
    <dataset>/meta.jsonl           every parameter per clip

ai-toolkit pairs a target with its control by the SAME file name (toolkit/dataloader_mixins.py at 4a7f285).

Parts:
- Poly Haven fetch (public API https://api.polyhaven.com: /assets?type=hdris&categories=, /files/{id} ->
  hdri.<res>.<fmt>.{url,size,md5}); resumable through ``.part`` + HTTP Range, size + md5 checked, skip existing.
- Hugging Face fetch: matching files listed and summed BEFORE anything downloads; refused above the GB cap.
- HDRI -> clips: equirect .hdr / .exr -> perspective renders (torch grid_sample) along seeded pan / tilt / push
  paths, written as half EXR frame folders.
- Make Pairs: control + target mp4, caption, meta.jsonl. Every encode / decode goes through mmx_utils/hdr_math.py.

Network access is injected (``Http`` / ``api`` arguments), so tests run with fakes and never touch the network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")          # before any cv2 import (EXR support)

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import hdr_math as HM
    from . import hdr_a29_jobs as J
except ImportError:                                              # imported as top-level mmx_utils.*
    from mmx_utils import hdr_math as HM
    from mmx_utils import hdr_a29_jobs as J

ENCODING = HM.DEFAULT_ENCODING                                   # 'diffhdr_log_16384' (fixed decision)
USER_AGENT = "C2C-MiniMaxSuite-HDR-dataset/1.1 (+https://github.com/Code2Collapse)"
POLYHAVEN_API = "https://api.polyhaven.com"
POLYHAVEN_RESOLUTIONS = ("1k", "2k", "4k", "8k")
POLYHAVEN_FORMATS = ("hdr", "exr")
# /categories/hdris on 2026-11-16 (one metadata GET); "all" = no filter. Collections are left out.
POLYHAVEN_CATEGORIES = ("all", "natural light", "outdoor", "nature", "urban", "low contrast", "high contrast",
                        "morning-afternoon", "partly cloudy", "indoor", "skies", "medium contrast", "clear",
                        "artificial light", "sunrise-sunset", "midday", "overcast", "studio", "night", "pure skies")
DRAFT_W, DRAFT_H = 1344, 768
DEFAULT_FRAMES = 73                                              # 17*4+5: 3.0 s at 24 fps
FPS = 24
MOTIONS = ("mix", "pan", "tilt", "push")
EXPOSURES = ("grey_0.18", "none")
CLIP_MODES = ("unclipped", "hard_clip", "soft_rolloff")
HALF_MAX = 65504.0
CHUNK = 1 << 20
DEFAULT_CAPTION_TEMPLATE = ("HDR scene-linear footage encoded in a log curve, high dynamic range. "
                            "Scene: {scene}. Lighting: {lighting}. Camera: {camera}.")
PAIRS_STATE = ".a29_pairs_state.json"
FETCH_STATE = ".a29_fetch_state.json"
CLIPS_STATE = ".a29_clips_state.json"


class FetchRefused(RuntimeError):
    """The fetch would exceed a limit (the GB cap) and was refused before downloading anything."""


# ====================================================================================================== frame grid
def check_frames(n: int) -> int:
    """H3 clips are 17k+5 frames (22, 39, 56, 73, 90 ...); anything else is refused with the nearest choices."""
    n = int(n)
    if n >= 22 and (n - 5) % 17 == 0:
        return n
    lo = max(22, (n - 5) // 17 * 17 + 5)
    raise ValueError(f"frames must be on H3's 17k+5 grid (22, 39, 56, 73, 90, 107 ...); {n} is not - use {lo} or "
                     f"{lo + 17}")


def check_side(n: int, what: str) -> int:
    if int(n) % 32 or int(n) < 64:
        raise ValueError(f"{what} must be a multiple of 32 (H3: 16x VAE x 2x2 patch), got {n}")
    return int(n)


# ====================================================================================================== HTTP layer
class Http:
    """The only network access of this module (urllib, a generic User-Agent, no personal data). Tests pass a fake
    with the same two methods."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 60.0):
        self.ua, self.timeout = user_agent, timeout

    def get_json(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def open_stream(self, url: str, start: int = 0):
        """(status, total_size_or_None, iterator of byte chunks). status 206 = the Range was honoured."""
        headers = {"User-Agent": self.ua}
        if start > 0:
            headers["Range"] = f"bytes={start}-"
        req = urllib.request.Request(url, headers=headers)
        r = urllib.request.urlopen(req, timeout=self.timeout)
        length = r.headers.get("Content-Length")

        def chunks():
            try:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    yield b
            finally:
                r.close()
        return r.status, (int(length) if length else None), chunks()


def md5_file(path: str | os.PathLike) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the API's md5, not security
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download(http, url: str, dest: str | os.PathLike, *, size: int | None = None, md5: str | None = None,
             job: "J.Job | None" = None) -> str:
    """Resumable download: bytes go to ``dest.part`` (continued with a Range request), then are checked (size, md5)
    and renamed. Returns 'skipped' (already complete), 'resumed' or 'downloaded'."""
    dest = Path(dest)
    if dest.is_file() and (size is None or dest.stat().st_size == size) and (md5 is None or md5_file(dest) == md5):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0
    status, _length, chunks = http.open_stream(url, have)
    mode = "ab"
    how = "resumed" if have else "downloaded"
    if have and status != 206:                                   # server ignored the Range: start over
        mode, have, how = "wb", 0, "downloaded"
    with open(part, mode) as f:
        for b in chunks:
            if job is not None:
                job.check_cancel()
            f.write(b)
    got = part.stat().st_size
    if size is not None and got != size:
        raise IOError(f"{dest.name}: got {got} bytes, the API says {size}; the .part file is kept, run again to resume")
    if md5 is not None and md5_file(part) != md5:
        part.unlink()
        raise IOError(f"{dest.name}: md5 mismatch after download; the partial file was deleted, run again")
    os.replace(part, dest)
    return how


# ====================================================================================================== Poly Haven
def polyhaven_list(http, category: str = "all", limit: int = 0) -> list[tuple[str, dict]]:
    """HDRI assets of a category, most downloaded first (ties by id), at most ``limit`` (0 = all)."""
    q = {"type": "hdris"}
    if category and category != "all":
        q["categories"] = category
    data = http.get_json(f"{POLYHAVEN_API}/assets?{urllib.parse.urlencode(q)}")
    if not isinstance(data, dict):
        raise ValueError("Poly Haven /assets did not return an object keyed by asset id")
    items = sorted(data.items(), key=lambda kv: (-int((kv[1] or {}).get("download_count") or 0), kv[0]))
    return items[: int(limit)] if limit and limit > 0 else items


def polyhaven_file(http, asset_id: str, resolution: str, fmt: str) -> dict:
    """{url, size, md5} of one HDRI file (``/files/{id}`` -> hdri -> resolution -> format)."""
    files = http.get_json(f"{POLYHAVEN_API}/files/{urllib.parse.quote(asset_id)}")
    try:
        return files["hdri"][resolution][fmt]
    except (KeyError, TypeError):
        have = sorted((files.get("hdri") or {}).keys()) if isinstance(files, dict) else []
        raise KeyError(f"{asset_id}: no {resolution} {fmt} file (available resolutions: {have})") from None


def fetch_polyhaven(out_dir: str, category: str = "all", resolution: str = "2k", limit: int = 20,
                    fmt: str = "hdr", *, http=None, job: "J.Job | None" = None) -> dict:
    """Download CC0 HDRIs into ``out_dir`` (+ one ``<id>.json`` sidecar each: name, categories, tags - the captions
    use them). Existing complete files are skipped; ``.part`` files resume."""
    if resolution not in POLYHAVEN_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {POLYHAVEN_RESOLUTIONS}")
    if fmt not in POLYHAVEN_FORMATS:
        raise ValueError(f"format must be one of {POLYHAVEN_FORMATS}")
    http = http or Http()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = J.StateFile(out / FETCH_STATE)
    assets = polyhaven_list(http, category, limit)
    rep = {"source": "polyhaven", "category": category, "resolution": resolution, "format": fmt,
           "requested": len(assets), "downloaded": 0, "resumed": 0, "skipped": 0, "failed": [], "files": []}
    for i, (aid, info) in enumerate(assets):
        if job is not None:
            job.check_cancel()
            job.set_progress(i, len(assets), f"{aid} ({i + 1}/{len(assets)})")
        dest = out / f"{aid}_{resolution}.{fmt}"
        key = f"{aid}:{resolution}:{fmt}"
        if state.is_done(key) and dest.is_file() and dest.stat().st_size == state.items[key].get("size", -1):
            rep["skipped"] += 1
            rep["files"].append(dest.name)
            continue
        try:
            meta = polyhaven_file(http, aid, resolution, fmt)
            how = download(http, meta["url"], dest, size=meta.get("size"), md5=meta.get("md5"), job=job)
        except J.JobCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad asset must not stop the rest; it is reported
            rep["failed"].append(f"{aid}: {exc}")
            if job is not None:
                job.log(f"FAILED {aid}: {exc}")
            continue
        rep[how if how in ("downloaded", "resumed", "skipped") else "downloaded"] += 1
        side = {"id": aid, "name": (info or {}).get("name", aid), "categories": (info or {}).get("categories", []),
                "tags": (info or {}).get("tags", []), "source": "polyhaven", "licence": "CC0",
                "url": f"https://polyhaven.com/a/{aid}", "file": dest.name}
        (out / f"{aid}_{resolution}.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
        state.mark(key, size=dest.stat().st_size, md5=meta.get("md5"))
        rep["files"].append(dest.name)
        if job is not None:
            job.log(f"{how} {dest.name}")
    if job is not None:
        job.set_progress(len(assets), len(assets), "finished")
    rep["folder"] = str(out)
    return rep


# ====================================================================================================== Hugging Face
def split_patterns(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,\n;]", text or "") if p.strip()]


def hf_matching_files(api, repo_id: str, patterns: list[str], *, repo_type: str = "dataset",
                      revision: str | None = None, token: str | None = None) -> list[tuple[str, int]]:
    """(path, size) of every file of the repo that matches the allow patterns (sizes from the Hub listing)."""
    from huggingface_hub.utils import filter_repo_objects
    entries = [e for e in api.list_repo_tree(repo_id, recursive=True, repo_type=repo_type, revision=revision or None,
                                             token=token) if getattr(e, "size", None) is not None
               and not hasattr(e, "tree_id")]
    keep = list(filter_repo_objects(entries, allow_patterns=patterns or None, key=lambda e: e.path))
    return [(e.path, int(e.size)) for e in keep]


def fetch_huggingface(repo_id: str, out_dir: str, allow_patterns: str = "*.exr, *.hdr", cap_gb: float = 5.0, *,
                      repo_type: str = "dataset", revision: str = "", api=None, downloader=None,
                      job: "J.Job | None" = None) -> dict:
    """snapshot_download of the matching files into ``out_dir``; the total size is checked first and the fetch is
    REFUSED above ``cap_gb``. HF_TOKEN comes from the environment only (never an input, never logged)."""
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must look like 'owner/name'")
    token = os.environ.get("HF_TOKEN") or None
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    patterns = split_patterns(allow_patterns)
    files = hf_matching_files(api, repo_id, patterns, repo_type=repo_type, revision=revision or None, token=token)
    total = sum(s for _, s in files)
    cap = float(cap_gb) * 1e9
    rep = {"source": "huggingface", "repo_id": repo_id, "repo_type": repo_type, "patterns": patterns,
           "files": len(files), "total_bytes": total, "total_gb": round(total / 1e9, 3), "cap_gb": cap_gb,
           "token": "HF_TOKEN from the environment" if token else "none (public repos only)"}
    if not files:
        raise FetchRefused(f"{repo_id}: no file matches {patterns}")
    if total > cap:
        raise FetchRefused(f"{repo_id}: the {len(files)} matching files total {total / 1e9:.2f} GB, above the cap of "
                           f"{cap_gb} GB - nothing was downloaded. Narrow allow_patterns or raise cap_gb.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = [(p, s) for p, s in files if not ((out / p).is_file() and (out / p).stat().st_size == s)]
    rep["skipped"] = len(files) - len(missing)
    if job is not None:
        job.set_progress(rep["skipped"], len(files), f"{len(missing)} files to fetch")
        job.check_cancel()
    if missing:
        if downloader is None:
            from huggingface_hub import snapshot_download as downloader
        kw = dict(repo_type=repo_type, local_dir=str(out), allow_patterns=patterns or None, token=token)
        if revision:
            kw["revision"] = revision
        if job is not None:
            kw["tqdm_class"] = _job_tqdm(job, rep["skipped"], len(files))
        downloader(repo_id, **kw)
    if job is not None:
        job.set_progress(len(files), len(files), "finished")
    rep["downloaded"] = len(missing)
    rep["folder"] = str(out)
    return rep


def _job_tqdm(job, base: int, total: int):
    """A tqdm class for snapshot_download that forwards file progress to the job and stops on Cancel."""
    from tqdm.auto import tqdm

    class JobTqdm(tqdm):
        def __init__(self, *a, **k):
            k["disable"] = False
            k.pop("name", None)
            super().__init__(*a, **k)

        def update(self, n=1):
            job.check_cancel()
            r = super().update(n)
            if self.total and self.total < 10000:                  # the per-file counter, not a byte bar
                job.set_progress(min(total, base + int(self.n)), total)
            return r
    return JobTqdm


# ====================================================================================================== HDRI -> clips
def load_hdri(path: str | os.PathLike) -> np.ndarray:
    """Equirect .hdr / .exr -> float32 [H, W, 3] linear RGB, unclamped."""
    path = str(path)
    if path.lower().endswith(".exr"):
        try:
            from . import hdr_a29_io as IO
        except ImportError:
            from mmx_utils import hdr_a29_io as IO
        return IO.read_exr(path)
    a = None
    try:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if a is not None:
            a = a[..., ::-1]
    except Exception:  # noqa: BLE001
        a = None
    if a is None:
        try:
            import OpenImageIO as oiio
            inp = oiio.ImageInput.open(path)
            if inp is None:
                raise RuntimeError(oiio.geterror())
            try:
                a = np.asarray(inp.read_image(oiio.FLOAT))[..., :3]
            finally:
                inp.close()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot read the HDRI {path}: {exc}") from exc
    return np.ascontiguousarray(a, dtype=np.float32)


def _ease(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def camera_path(motion: str, frames: int, rng: random.Random, fov_deg: float = 60.0) -> dict:
    """Per-frame yaw / pitch / horizontal FOV (degrees) for one clip. pan: yaw sweeps 20..60 deg; tilt: pitch sweeps
    8..20 deg; push: the FOV narrows to 60..80 % (a zoom-in). Start yaw is random, start pitch -10..+15 deg."""
    if motion == "mix":
        motion = rng.choice(("pan", "tilt", "push"))
    if motion not in ("pan", "tilt", "push"):
        raise ValueError(f"motion must be one of {MOTIONS}")
    t = _ease(np.linspace(0.0, 1.0, frames, dtype=np.float64))
    yaw0, pitch0 = rng.uniform(-180.0, 180.0), rng.uniform(-10.0, 15.0)
    yaw = np.full(frames, yaw0)
    pitch = np.full(frames, pitch0)
    fov = np.full(frames, float(fov_deg))
    if motion == "pan":
        span = rng.uniform(20.0, 60.0) * rng.choice((-1.0, 1.0))
        yaw = yaw0 + span * t
        desc = f"slow pan {'right' if span > 0 else 'left'}"
        amount = span
    elif motion == "tilt":
        span = rng.uniform(8.0, 20.0) * rng.choice((-1.0, 1.0))
        pitch = np.clip(pitch0 + span * t, -60.0, 60.0)
        desc = f"slow tilt {'up' if span > 0 else 'down'}"
        amount = span
    else:
        k = rng.uniform(0.6, 0.8)
        fov = fov_deg * (1.0 + (k - 1.0) * t)
        desc = "slow push in"
        amount = k
    return {"motion": motion, "yaw": yaw.tolist(), "pitch": pitch.tolist(), "fov": fov.tolist(),
            "description": desc, "amount": float(amount), "yaw0": yaw0, "pitch0": pitch0}


def render_perspective(eq: torch.Tensor, yaw_deg: float, pitch_deg: float, fov_deg: float, width: int,
                       height: int) -> torch.Tensor:
    """One pinhole view of an equirect panorama [H, W, 3] (lon -180..180 left to right, lat +90 top), bilinear,
    seam-safe (circular padding). yaw / pitch = where the image centre looks. Returns [height, width, 3]."""
    dev = eq.device
    H, W = eq.shape[0], eq.shape[1]
    f = math.tan(math.radians(fov_deg) / 2.0)
    xs = ((torch.arange(width, device=dev, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0) * f
    ys = -((torch.arange(height, device=dev, dtype=torch.float32) + 0.5) / height * 2.0 - 1.0) * f * height / width
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gz = torch.ones_like(gx)
    p, yw = math.radians(pitch_deg), math.radians(yaw_deg)
    y1 = gy * math.cos(p) + gz * math.sin(p)
    z1 = -gy * math.sin(p) + gz * math.cos(p)
    x2 = gx * math.cos(yw) + z1 * math.sin(yw)
    z2 = -gx * math.sin(yw) + z1 * math.cos(yw)
    n = torch.sqrt(x2 * x2 + y1 * y1 + z2 * z2)
    lon = torch.atan2(x2, z2)
    lat = torch.asin((y1 / n).clamp(-1.0, 1.0))
    pad = 2
    src = torch.cat([eq[:, -pad:], eq, eq[:, :pad]], dim=1).permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W+2p]
    px = (lon / (2 * math.pi) + 0.5) * W - 0.5 + pad                                        # pixel centres
    py = (0.5 - lat / math.pi) * H - 0.5
    grid = torch.stack([(px + 0.5) / (W + 2 * pad) * 2.0 - 1.0, (py + 0.5) / H * 2.0 - 1.0], dim=-1)
    out = F.grid_sample(src, grid.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0)


def normalise_exposure(frames: torch.Tensor, mode: str) -> tuple[torch.Tensor, float]:
    """grey_0.18: the median Rec.709 luma of ``frames`` becomes 0.18 (diffuse grey, so 1.0 = diffuse white);
    none: as is. Returns (frames * scale, scale)."""
    if mode == "none":
        return frames, 1.0
    if mode != "grey_0.18":
        raise ValueError(f"exposure must be one of {EXPOSURES}")
    y = HM.luma709(frames)
    med = float(torch.quantile(y.flatten()[:: max(1, y.numel() // 1_000_000)].float(), 0.5))
    s = 0.18 / max(med, 1e-6)
    return frames * s, s


def hdri_meta(path: Path) -> dict:
    """Name / categories / tags of an HDRI from its Poly Haven sidecar (``<stem>.json``), else from the file name."""
    side = path.with_suffix(".json")
    if side.is_file():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    stem = re.sub(r"_(1k|2k|4k|8k|16k)$", "", path.stem)
    return {"id": stem, "name": stem.replace("_", " ").strip().title(), "categories": [], "tags": []}


def list_hdris(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".hdr", ".exr") and p.is_file())


def write_exr_frame(path: Path, rgb: torch.Tensor) -> None:
    try:
        from . import hdr_a29_io as IO
    except ImportError:
        from mmx_utils import hdr_a29_io as IO
    IO.write_exr(str(path), rgb.clamp(0.0, HALF_MAX).cpu().numpy(), precision="half")


def read_exr_frame(path: Path) -> np.ndarray:
    try:
        from . import hdr_a29_io as IO
    except ImportError:
        from mmx_utils import hdr_a29_io as IO
    return IO.read_exr(str(path))


def hdri_to_clips(hdri_folder: str, out_dir: str, *, clips_per_hdri: int = 2, frames: int = DEFAULT_FRAMES,
                  width: int = DRAFT_W, height: int = DRAFT_H, fov_deg: float = 60.0, motion: str = "mix",
                  exposure: str = "grey_0.18", seed: int = 0, limit: int = 0, job: "J.Job | None" = None,
                  writer: Callable[[Path, torch.Tensor], None] | None = None) -> dict:
    """Render ``clips_per_hdri`` camera moves per HDRI into ``out_dir/<clip>/frame_00000.exr`` + ``clip.json``.
    Finished clips (clip.json complete) are skipped. Linear values above 1 are kept (half EXR caps at 65504)."""
    check_frames(frames)
    check_side(width, "width")
    check_side(height, "height")
    if motion not in MOTIONS:
        raise ValueError(f"motion must be one of {MOTIONS}")
    hdris = list_hdris(hdri_folder)
    if limit and limit > 0:
        hdris = hdris[:limit]
    if not hdris:
        raise FileNotFoundError(f"no .hdr / .exr files in {hdri_folder}")
    writer = writer or write_exr_frame
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = J.StateFile(out / CLIPS_STATE)
    total = len(hdris) * int(clips_per_hdri)
    rep = {"hdris": len(hdris), "clips": 0, "skipped": 0, "frames": frames, "size": [width, height], "clip_ids": []}
    k = 0
    for path in hdris:
        meta = hdri_meta(path)
        eq_t = None
        for c in range(int(clips_per_hdri)):
            clip_id = f"{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)}_c{c:02d}"
            if job is not None:
                job.check_cancel()
                job.set_progress(k, total, clip_id)
            k += 1
            folder = out / clip_id
            if state.is_done(clip_id) and (folder / "clip.json").is_file() \
                    and len(list(folder.glob("frame_*.exr"))) == frames:
                rep["skipped"] += 1
                rep["clip_ids"].append(clip_id)
                continue
            if eq_t is None:
                eq_t = torch.from_numpy(load_hdri(path))
            rng = random.Random(f"{seed}:{path.stem}:{c}")
            cam = camera_path(motion, frames, rng, fov_deg)
            # exposure from 5 quarter-size probe views along the path (the whole clip is never held in RAM)
            probe_idx = sorted({round(x) for x in np.linspace(0, frames - 1, 5)})
            probe = torch.stack([render_perspective(eq_t, cam["yaw"][i], cam["pitch"][i], cam["fov"][i],
                                                    max(32, width // 4), max(32, height // 4)) for i in probe_idx])
            _, scale = normalise_exposure(probe, exposure)
            del probe
            folder.mkdir(parents=True, exist_ok=True)
            peak, above = 0.0, 0
            for i in range(frames):
                if job is not None:
                    job.check_cancel()
                view = render_perspective(eq_t, cam["yaw"][i], cam["pitch"][i], cam["fov"][i], width, height) * scale
                peak = max(peak, float(view.max()))
                above += int((view > HM.enc_max(ENCODING)).sum())
                writer(folder / f"frame_{i:05d}.exr", view)
            info = {"clip_id": clip_id, "hdri": path.name, "hdri_meta": meta, "frames": frames, "fps": FPS,
                    "size": [width, height], "seed": seed, "camera": cam, "exposure": exposure,
                    "exposure_scale": scale, "peak_linear": peak,
                    "share_above_16384": above / float(frames * width * height * 3),
                    "space": "scene-linear Rec.709, 1.0 = diffuse white", "complete": True}
            (folder / "clip.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            state.mark(clip_id, frames=frames)
            rep["clips"] += 1
            rep["clip_ids"].append(clip_id)
            if job is not None:
                job.log(f"{clip_id}: {cam['description']}, peak {peak:.1f}x white")
        eq_t = None
    if job is not None:
        job.set_progress(total, total, "finished")
    rep["folder"] = str(out)
    return rep


# ====================================================================================================== pairs
def ffmpeg_exe() -> str:
    try:
        from . import hdr_a29_io as IO
    except ImportError:
        from mmx_utils import hdr_a29_io as IO
    return IO.ffmpeg_exe()


def assign_modes(clip_ids: Iterable[str], seed: int, unclipped_share: float = 0.15) -> dict:
    """Clip mode per clip: exactly round(share * N) clips 'unclipped', the rest split half 'hard_clip' / half
    'soft_rolloff'. The order is a seeded hash of the id, so a clip keeps its mode when others are added."""
    ids = sorted(set(clip_ids), key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
    n_un = int(round(float(unclipped_share) * len(ids)))
    out = {}
    for i, cid in enumerate(ids):
        out[cid] = "unclipped" if i < n_un else ("hard_clip" if (i - n_un) % 2 == 0 else "soft_rolloff")
    return out


def sdr_control(lin: torch.Tensor, mode: str, ev: float, peak: float | None = None) -> tuple[torch.Tensor, dict]:
    """Scene-linear [T, H, W, 3] -> display sRGB code 0..1 (what an SDR camera / grade would deliver).
    hard_clip: exposure then clip at white; soft_rolloff: exposure then a soft shoulder (hdr_math.tonemap_preview,
    knee 0.5); unclipped: exposure lowered until the clip's peak (``peak``, default: of ``lin``) fits under white,
    so nothing clips. Pass ``peak`` when ``lin`` is one frame of a longer clip."""
    if mode == "unclipped":
        peak = float(lin.max()) if peak is None else float(peak)
        ev_used = min(float(ev), -math.log2(max(peak, 1e-12)))
        x = lin * (2.0 ** ev_used)
        return HM.linear_to_srgb(x), {"ev": ev_used, "ev_drawn": ev, "clipped_share": float((x > 1.0).float().mean())}
    if mode == "hard_clip":
        x = lin * (2.0 ** ev)
        return HM.tonemap_preview(lin, ev, shoulder=0.0), {"ev": ev, "clipped_share": float((x > 1.0).float().mean())}
    if mode == "soft_rolloff":
        x = lin * (2.0 ** ev)
        return HM.tonemap_preview(lin, ev, shoulder=0.5), {"ev": ev, "knee": 0.5,
                                                           "above_knee_share": float((x > 0.5).float().mean())}
    raise ValueError(f"mode must be one of {CLIP_MODES}")


def to_u8(code01: torch.Tensor) -> np.ndarray:
    return (code01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


def write_mp4(path: Path, frames_u8, *, lossless_rgb: bool, crf: int = 18, fps: int = FPS,
              job: "J.Job | None" = None, size: tuple[int, int] | None = None) -> list[str]:
    """uint8 RGB frames ([T, H, W, 3] array, or an iterable of [H, W, 3] with ``size`` = (w, h)) -> mp4, streamed
    into ffmpeg. lossless_rgb: libx264rgb -crf 0 rgb24 (bit-exact); else libx264 yuv420p at ``crf``."""
    if isinstance(frames_u8, np.ndarray):
        h, w = frames_u8.shape[1], frames_u8.shape[2]
    elif size is not None:
        w, h = size
    else:
        raise ValueError("pass size=(w, h) with an iterable of frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    argv = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if lossless_rgb:
        argv += ["-c:v", "libx264rgb", "-crf", "0", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "rgb24"]
    else:
        argv += ["-c:v", "libx264", "-crf", str(int(crf)), "-preset", "medium", "-pix_fmt", "yuv420p",
                 "-color_primaries", "bt709", "-color_trc", "iec61966-2-1", "-colorspace", "bt709",
                 "-color_range", "tv"]
    argv += ["-an", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in frames_u8:
            if job is not None and job.cancelled:
                proc.kill()
                proc.wait()
                tmp.unlink(missing_ok=True)
                job.check_cancel()
            proc.stdin.write(np.ascontiguousarray(fr).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    except BrokenPipeError:
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed writing {path.name}: {err.strip()[-500:]}")
    os.replace(tmp, path)
    return argv


def read_mp4_rgb(path: Path, width: int, height: int) -> np.ndarray:
    """mp4 -> [T, H, W, 3] uint8 RGB through ffmpeg (used by tests and the pair check)."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, height, width, 3)


_LIGHTING = ("night", "overcast", "sunrise-sunset", "midday", "morning-afternoon", "studio", "artificial light",
             "natural light", "clear", "partly cloudy", "high contrast", "low contrast", "medium contrast")


def caption_fields(info: dict) -> dict:
    """scene / lighting / camera for a clip, derived from the HDRI's name + Poly Haven categories / tags and the
    camera move (no vision model: honest but generic captions)."""
    meta = info.get("hdri_meta") or {}
    cats = [str(c) for c in meta.get("categories") or []]
    tags = [str(t) for t in meta.get("tags") or []]
    place = [c for c in cats if c in ("outdoor", "indoor", "urban", "nature", "skies", "pure skies", "studio")]
    scene = meta.get("name") or info.get("hdri", "an HDRI")
    extra = ", ".join((place + tags)[:4])
    scene = f"{scene} ({extra})" if extra else str(scene)
    light = [c for c in _LIGHTING if c in cats]
    lighting = ", ".join(light[:3]) if light else "natural high dynamic range lighting"
    cam = info.get("camera") or {}
    camera = f"{cam.get('description', 'static camera')}, {info.get('size', [DRAFT_W, DRAFT_H])[0]}x" \
             f"{info.get('size', [DRAFT_W, DRAFT_H])[1]}, {info.get('fps', FPS)} fps"
    return {"scene": scene, "lighting": lighting, "camera": camera}


def render_caption(template: str, fields: dict) -> str:
    try:
        return (template or DEFAULT_CAPTION_TEMPLATE).format(**fields).strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"the caption template may only use {{scene}}, {{lighting}} and {{camera}}: {exc}") from exc


def list_clips(clips_folder: str) -> list[Path]:
    out = []
    for d in sorted(Path(clips_folder).iterdir()) if Path(clips_folder).is_dir() else []:
        j = d / "clip.json"
        if d.is_dir() and j.is_file():
            try:
                if json.loads(j.read_text(encoding="utf-8")).get("complete"):
                    out.append(d)
            except (OSError, ValueError):
                pass
    return out


def make_pairs(clips_folder: str, out_dir: str, *, seed: int = 0, ev_range: float = 1.5,
               unclipped_share: float = 0.15, crf_min: int = 16, crf_max: int = 28,
               caption_template: str = DEFAULT_CAPTION_TEMPLATE, fps: int = FPS, job: "J.Job | None" = None,
               reader: Callable[[Path], np.ndarray] | None = None) -> dict:
    """EXR clips -> the ai-toolkit pair layout (module docstring). Finished pairs are skipped."""
    if not (0 <= crf_min <= crf_max <= 51):
        raise ValueError("need 0 <= crf_min <= crf_max <= 51")
    if not (0.0 <= unclipped_share <= 1.0):
        raise ValueError("unclipped_share must be 0..1")
    reader = reader or read_exr_frame
    clips = list_clips(clips_folder)
    if not clips:
        raise FileNotFoundError(f"no finished clips (folders with clip.json) in {clips_folder} - run HDRI -> Clips")
    out = Path(out_dir)
    tdir, cdir = out / "targets", out / "control"
    tdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    state = J.StateFile(out / PAIRS_STATE)
    modes = assign_modes([c.name for c in clips], seed, unclipped_share)
    meta_path = out / "meta.jsonl"
    old = {}
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                old[row["clip_id"]] = row
            except (ValueError, KeyError):
                pass
    rep = {"pairs": 0, "skipped": 0, "modes": {m: 0 for m in CLIP_MODES}, "frames": set()}
    for i, folder in enumerate(clips):
        cid = folder.name
        if job is not None:
            job.check_cancel()
            job.set_progress(i, len(clips), cid)
        rep["modes"][modes[cid]] += 1
        files = (tdir / f"{cid}.mp4", tdir / f"{cid}.txt", cdir / f"{cid}.mp4")
        if state.is_done(cid) and all(f.is_file() for f in files) and cid in old \
                and old[cid].get("clip_mode") == modes[cid]:
            rep["skipped"] += 1
            rep["frames"].add(old[cid]["frames"])
            continue
        info = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
        frame_files = sorted(folder.glob("frame_*.exr"))
        n = check_frames(len(frame_files))
        # streamed frame by frame (a 73-frame 1344x768 clip is ~0.9 GB as float32): pass 1 = peak, then one pass
        # per video
        peak, above, first = 0.0, 0, None
        for p in frame_files:
            fr = torch.from_numpy(reader(p)).float().clamp_min(0.0)
            first = first if first is not None else fr.shape
            peak = max(peak, float(fr.max()))
            above += int((fr > HM.enc_max(ENCODING)).sum())
        h, w = first[0], first[1]
        rng = random.Random(f"{seed}:{cid}:pairs")
        ev = rng.uniform(-float(ev_range), float(ev_range))
        crf = rng.randint(int(crf_min), int(crf_max))
        cinfo_box: dict = {}

        def control_frames():
            for p in frame_files:
                fr = torch.from_numpy(reader(p)).float().clamp_min(0.0)
                sdr, ci = sdr_control(fr, modes[cid], ev, peak=peak)
                cinfo_box.setdefault("first", ci)
                cinfo_box["clipped"] = cinfo_box.get("clipped", 0.0) + ci.get("clipped_share",
                                                                             ci.get("above_knee_share", 0.0))
                yield to_u8(sdr)

        def target_frames():
            for p in frame_files:
                yield to_u8(HM.encode(torch.from_numpy(reader(p)).float().clamp_min(0.0), ENCODING))

        write_mp4(cdir / f"{cid}.mp4", control_frames(), lossless_rgb=False, crf=crf, fps=fps, job=job, size=(w, h))
        write_mp4(tdir / f"{cid}.mp4", target_frames(), lossless_rgb=True, fps=fps, job=job, size=(w, h))
        cinfo = dict(cinfo_box["first"])
        share_key = "clipped_share" if "clipped_share" in cinfo else "above_knee_share"
        cinfo[share_key] = cinfo_box["clipped"] / n
        fields = caption_fields(info)
        caption = render_caption(caption_template, fields)
        (tdir / f"{cid}.txt").write_text(caption + "\n", encoding="utf-8")
        row = {"clip_id": cid, "frames": n, "fps": fps, "size": [w, h], "seed": seed, "clip_mode": modes[cid],
               **{f"control_{k}": v for k, v in cinfo.items()}, "control_crf": crf,
               "control_codec": "libx264 yuv420p 8-bit sRGB (bt709 / iec61966-2-1)",
               "target_encoding": ENCODING, "target_codec": "libx264rgb crf 0 rgb24 (lossless)",
               "target_quantisation": "8-bit code values: the log code is quantised to 1/255 before the lossless "
                                      "encode (ai-toolkit reads video as 8-bit)",
               "peak_linear": peak, "share_above_range": above / float(n * h * w * 3),
               "caption_fields": fields, "caption": caption, "source": info.get("hdri"),
               "camera": (info.get("camera") or {}).get("description"),
               "files": {"target": f"targets/{cid}.mp4", "control": f"control/{cid}.mp4",
                         "caption": f"targets/{cid}.txt"}}
        old[cid] = row
        meta_path.write_text("".join(json.dumps(old[k]) + "\n" for k in sorted(old)), encoding="utf-8")
        state.mark(cid, frames=n, mode=modes[cid])
        rep["pairs"] += 1
        rep["frames"].add(n)
        if job is not None:
            job.log(f"{cid}: {modes[cid]}, ev {cinfo['ev']:+.2f}, crf {crf}")
    if job is not None:
        job.set_progress(len(clips), len(clips), "finished")
    rep["frames"] = sorted(rep["frames"])
    rep.update(folder=str(out), targets=str(tdir), control=str(cdir), meta=str(meta_path), clips=len(clips))
    return rep
