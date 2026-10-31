"""H3 Shot Publisher (owner A21 Revision 3 Part 9.10; ledger L9.74 slice 10).

Publishes rendered media (a file, or a folder such as an image sequence) under
``<output>/<publish_root>/<show>/<seq>/<shot>/<task>/v###/`` with a name built from a template, a stage (wip / review /
final) and a manifest:

* the version is RESERVED atomically - its folder is created with ``os.mkdir``, which fails if it exists - so two
  publishes at the same moment never take the same version;
* ``require_qc = pass`` refuses to publish without a passing QC verdict;
* a published version is never overwritten unless ``overwrite = same version`` names that version explicitly;
* every copy is verified (sha256 of the copy == sha256 of the source) when ``checksum`` is on;
* every path stays inside ComfyUI's output folder; names are checked (no folders, no ``..``).

Original node, GPL-3.0. Design: docs/evidence/A21/phase1/DESIGN_slices7_10.md in the workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import string
import time
from pathlib import Path

from comfy_api.latest import io

from .review_package import qc_summary

WHO = "H3 Shot Publisher"
CONTRACT = "h3.publish_manifest"
STAGES = ("wip", "review", "final")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VDIR = re.compile(r"^v(\d{3,})$")
DEFAULT_TEMPLATE = "{show}_{seq}_{shot}_{task}_{stage}_v{version:03d}{ext}"


def _output_root() -> Path:
    import folder_paths
    return Path(folder_paths.get_output_directory()).resolve()


def _inside(root: Path, p: Path) -> bool:
    return p == root or root in p.parents


def _check_name(label: str, value: str) -> str:
    v = str(value or "").strip()
    if not _NAME.match(v) or ".." in v:
        raise ValueError(f"{WHO}: {label} {value!r} - use letters, digits, '.', '_' or '-' (no folders, no '..').")
    return v


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


def resolve_media(media_path: str) -> tuple[Path, list[Path]]:
    """(the media path, its files) - inside the output folder only."""
    root = _output_root()
    raw = str(media_path or "").strip().strip('"')
    if not raw:
        raise ValueError(f"{WHO}: 'media_path' is empty - give a file or folder inside ComfyUI's output folder.")
    p = Path(raw)
    p = (p if p.is_absolute() else root / p).resolve()
    if not _inside(root, p):
        raise ValueError(f"{WHO}: {raw!r} is outside ComfyUI's output folder ({root}); publish from inside it.")
    if p.is_file():
        return p, [p]
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))
        if not files:
            raise ValueError(f"{WHO}: the folder {p} is empty.")
        return p, files
    raise ValueError(f"{WHO}: nothing found at {p}.")


def reserve_version(shot_dir: Path, version: int, overwrite: str) -> tuple[int, Path, bool]:
    """(version, its folder, reused). 0 = the next free version, reserved with os.mkdir (atomic)."""
    shot_dir.mkdir(parents=True, exist_ok=True)
    if int(version) > 0:
        vdir = shot_dir / f"v{int(version):03d}"
        try:
            os.mkdir(vdir)
            return int(version), vdir, False
        except FileExistsError:
            if overwrite != "same version":
                raise ValueError(f"{WHO}: version v{int(version):03d} is already published - pick another version, "
                                 f"use 0 (the next free one), or set overwrite = same version.") from None
            return int(version), vdir, True
    for _ in range(1000):
        taken = [int(m.group(1)) for d in shot_dir.iterdir() if d.is_dir() and (m := _VDIR.match(d.name))]
        nxt = (max(taken) + 1) if taken else 1
        vdir = shot_dir / f"v{nxt:03d}"
        try:
            os.mkdir(vdir)                                   # atomic: a concurrent publish gets FileExistsError
            return nxt, vdir, False
        except FileExistsError:
            continue
    raise ValueError(f"{WHO}: could not reserve a version in {shot_dir}")


def _render(template: str, fields: dict) -> str:
    try:
        name = string.Formatter().vformat(template, (), fields)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"{WHO}: naming template {template!r}: {exc} - fields are show, seq, shot, task, stage, "
                         f"version, ext") from exc
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"{WHO}: the naming template made {name!r} - a name, not a path.")
    return name


class MiniMaxH3_ShotPublisher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_ShotPublisher",
            display_name="H3 Shot Publisher",
            category="MiniMax H3/Media",
            description=(
                "Publishes rendered media into <output>/<publish_root>/<show>/<seq>/<shot>/<task>/v###/ with a "
                "reserved version, a stage, optional required QC, verified checksums and a manifest. Never "
                "overwrites another version; never writes outside the output folder."
            ),
            inputs=[
                io.String.Input("media_path", default="", tooltip="A file or folder inside ComfyUI's output folder "
                                                                 "(e.g. the Sequence Assembler's output)."),
                io.String.Input("show", default="show"),
                io.String.Input("seq", default="seq"),
                io.String.Input("shot", default="sh010"),
                io.String.Input("task", default="comp"),
                io.Combo.Input("stage", options=list(STAGES), default="wip"),
                io.String.Input("naming", default=DEFAULT_TEMPLATE,
                                tooltip="Fields: {show} {seq} {shot} {task} {stage} {version:03d} {ext} (a folder "
                                        "is published as a folder of that name)."),
                io.Int.Input("version", default=0, min=0, max=9999, tooltip="0 = the next free version."),
                io.String.Input("qc", default="", optional=True, tooltip="QC JSON (as H3 Review Package reads it)."),
                io.Combo.Input("require_qc", options=["none", "pass"], default="none"),
                io.Combo.Input("overwrite", options=["never", "same version"], default="never"),
                io.Boolean.Input("checksum", default=True, tooltip="sha256 every file and verify the copies."),
                io.String.Input("publish_root", default="publish", tooltip="Inside ComfyUI's output folder."),
                io.String.Input("metadata", default="", optional=True, tooltip="Extra JSON kept in the manifest."),
            ],
            outputs=[
                io.String.Output("published_path"),
                io.String.Output("manifest", tooltip="h3.publish_manifest v1."),
                io.String.Output("delivery_paths", tooltip="JSON list of every published file."),
                io.String.Output("status", tooltip="The publish checklist."),
            ],
        )

    @classmethod
    def execute(cls, media_path="", show="show", seq="seq", shot="sh010", task="comp", stage="wip",
                naming=DEFAULT_TEMPLATE, version=0, qc="", require_qc="none", overwrite="never", checksum=True,
                publish_root="publish", metadata="") -> io.NodeOutput:
        if stage not in STAGES:
            raise ValueError(f"{WHO}: stage {stage!r} - one of {', '.join(STAGES)}")
        fields = {k: _check_name(k, v) for k, v in (("show", show), ("seq", seq), ("shot", shot), ("task", task))}
        src, files = resolve_media(media_path)
        qcs = qc_summary(json.loads(qc) if str(qc or "").strip() else None)
        checks = [f"[x] media: {len(files)} file(s) at {src}"]
        if require_qc == "pass":
            if qcs["verdict"] != "pass":
                failed = [c["name"] for c in qcs["checks"] if not c["ok"]]
                raise ValueError(f"{WHO}: require_qc = pass, and the QC verdict is {qcs['verdict']}"
                                 + (f" (failed: {', '.join(failed)})" if failed else " (no QC connected)")
                                 + " - nothing was published.")
            checks.append("[x] QC passed")
        else:
            checks.append(f"[-] QC not required (verdict: {qcs['verdict']})")
        root = _output_root()
        proot = (root / str(publish_root or "publish").strip().strip("/\\")).resolve()
        if not _inside(root, proot):
            raise ValueError(f"{WHO}: publish_root {publish_root!r} points outside ComfyUI's output folder.")
        shot_dir = proot / fields["show"] / fields["seq"] / fields["shot"] / fields["task"]
        ver, vdir, reused = reserve_version(shot_dir, int(version), overwrite)
        checks.append(f"[x] version v{ver:03d} {'re-published (overwrite = same version)' if reused else 'reserved'}")
        ext = src.suffix if src.is_file() else ""
        name = _render(str(naming or DEFAULT_TEMPLATE), dict(fields, stage=stage, version=ver, ext=ext))
        target = vdir / name
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        records, missing = [], []
        try:
            for f in files:
                dest = target / f.name if src.is_dir() else target
                shutil.copy2(f, dest)
                rec = {"source": str(f), "path": str(dest), "bytes": dest.stat().st_size}
                if checksum:
                    s1, s2 = _sha256(f), _sha256(dest)
                    if s1 != s2:
                        raise ValueError(f"{WHO}: the copy of {f.name} does not match its source (sha256) - "
                                         f"the publish was rolled back.")
                    rec["sha256"] = s2
                records.append(rec)
        except Exception:
            if not reused:
                shutil.rmtree(vdir, ignore_errors=True)       # a failed publish leaves no half version
            raise
        for f in files:
            if not f.exists():
                missing.append(str(f))
        checks.append(f"[x] {len(records)} file(s) copied" + (", sha256 verified" if checksum else ""))
        if missing:
            checks.append(f"[ ] missing after copy: {', '.join(missing)}")
        manifest = {
            "c2c_contract": CONTRACT, "c2c_contract_version": "1.0", "published": str(target),
            "version": ver, "stage": stage, **fields, "source": str(src), "files": records, "qc": qcs,
            "metadata": json.loads(metadata) if str(metadata or "").strip() else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reused_version": reused,
        }
        (vdir / f"{Path(name).stem or name}.publish.json").write_text(json.dumps(manifest, indent=1),
                                                                         encoding="utf-8")
        status = "\n".join([f"published {stage} v{ver:03d}: {target}"] + checks)
        return io.NodeOutput(str(target), json.dumps(manifest), json.dumps([r["path"] for r in records]), status)


NODE_CLASS_MAPPINGS = {"MiniMaxH3_ShotPublisher": MiniMaxH3_ShotPublisher}
