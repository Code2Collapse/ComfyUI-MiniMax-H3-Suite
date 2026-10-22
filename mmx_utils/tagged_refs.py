# PORTED FROM: https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop@43afeff chain_nodes.py
# (_normalize_reference_tag, _append_scheduled_reference / _append_tagged_reference, _active_reference_bindings,
# _replace_reference_aliases, _compile_scheduled_reference_prompt, _tensor_fingerprint), GPL-3.0,
# Copyright (C) 2026 NikoDemon80 / ethanfel. Kept: the tag grammar, strict / soft / disabled compliance, the native
# numbering (pictures, then each video with its soundtrack's <Audio j> first, then standalone audio) and H3's 9 / 3 / 3
# capacity. Changed: no scenes or Plan (the prompt alone activates a reference; the suite's H3 Continuation runs the
# segments), numbering follows the chain order the user built (upstream sorts each kind by tag name although its
# tooltips promise chain order), no Qwen-only #anchors, caches, RefMod backend or fingerprint lineage.
"""Tagged references: register pictures, videos and audio under @tags, then write @tags in the prompt.

Only the references a prompt mentions are sent to H3, numbered the way core's Reference to Video numbers them, and the
@tags are replaced by those native labels (<Picture 1>, <Video 1>, <Audio 1>, ...).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import torch

REGISTRY_KIND = "mmx_tagged_references"
VERSION = 1
TAG_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
ALIAS_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,63})")
COMPLIANCE = ("strict", "soft", "disabled")
CAPACITY = {"picture": 9, "video": 3, "audio": 3}       # core MiniMaxH3ReferenceToVideo autogrow maxima
KINDS = ("picture", "video", "audio")


class TaggedRefError(ValueError):
    pass


def normalize_tag(value: Any, label: str = "Reference tag") -> str:
    tag = str(value or "").strip()
    if tag.startswith("@"):
        tag = tag[1:]
    if TAG_RE.fullmatch(tag) is None:
        raise TaggedRefError(f"{label} must be a stable tag such as 'hero_face' or '@hero-face' (a letter first, then "
                             f"letters, digits, '_' or '-', at most 64); got {value!r}.")
    return tag


def tensor_fingerprint(value: torch.Tensor) -> str:
    """sha256 over shape, dtype and bytes, in chunks (no giant bytes object)."""
    d = hashlib.sha256()
    d.update(str(tuple(int(s) for s in value.shape)).encode("ascii"))
    d.update(str(value.dtype).encode("ascii"))
    for chunk in (value.detach().split(8, dim=0) if value.ndim else (value.detach(),)):
        cpu = chunk.to(device="cpu").contiguous()
        if cpu.numel():
            d.update(memoryview(cpu.float().numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()).cast("B"))
    return d.hexdigest()


def audio_fingerprint(audio: dict) -> str:
    return tensor_fingerprint(audio["waveform"]) + f":{int(audio['sample_rate'])}"


def validate_audio(audio: Any, label: str) -> None:
    if (not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")) or audio["waveform"].ndim != 3
            or int(audio.get("sample_rate") or 0) <= 0):
        raise TaggedRefError(f"{label} must be a ComfyUI AUDIO ({{'waveform': [B, channels, samples], "
                             "'sample_rate': int}}).")
    if int(audio["waveform"].shape[-1]) == 0:
        raise TaggedRefError(f"{label} is empty.")


def entries_of(registry: Any) -> list[dict]:
    if registry is None:
        return []
    if not isinstance(registry, dict) or registry.get("kind") != REGISTRY_KIND:
        raise TaggedRefError("Tagged references must come from H3 Tagged Picture / Video / Audio nodes.")
    return list(registry["entries"])


def entry_tags(entry: dict) -> tuple[str, ...]:
    return (entry["tag"],) + ((entry["audio_tag"],) if entry.get("audio_tag") else ())


def make_registry(entries: list[dict]) -> dict:
    d = hashlib.sha256()
    for e in entries:
        d.update(f"{e['kind']}|{e['tag']}|{e.get('audio_tag', '')}|{e['content_hash']}|"
                 f"{e.get('audio_hash', '')};".encode("utf-8"))
    return {"kind": REGISTRY_KIND, "version": VERSION, "entries": entries, "fingerprint": d.hexdigest()}


def append(previous: Any, kind: str, tag: Any, value: Any, content_hash: str, audio: Any = None, audio_tag: Any = "",
           audio_hash: str = "") -> dict:
    """A new registry with one more entry. Tags are unique across the chain (a video's soundtrack tag included)."""
    if kind not in KINDS:
        raise TaggedRefError(f"unknown reference kind {kind!r}")
    entries = entries_of(previous)
    entry = {"kind": kind, "tag": normalize_tag(tag), "value": value, "content_hash": str(content_hash)}
    if audio is not None:
        entry.update({"audio": audio, "audio_tag": normalize_tag(audio_tag or entry["tag"] + "_audio",
                                                                 "Soundtrack tag"),
                      "audio_hash": str(audio_hash)})
        if entry["audio_tag"] == entry["tag"]:
            raise TaggedRefError("A video's @tag and its soundtrack's @audio_tag must differ.")
    taken = {t for e in entries for t in entry_tags(e)}
    clash = sorted(taken.intersection(entry_tags(entry)))
    if clash:
        raise TaggedRefError(f"@{clash[0]} is already registered in this chain; every tag must be unique.")
    return make_registry(entries + [entry])


def bindings(registry: Any, prompt: str, compliance: str = "strict") -> dict:
    """Which references the prompt activates, their native labels, and what was dropped or unknown."""
    if compliance not in COMPLIANCE:
        raise TaggedRefError(f"compliance must be one of {', '.join(COMPLIANCE)}; got {compliance!r}")
    entries = entries_of(registry)
    mentioned = set(ALIAS_RE.findall(str(prompt or "")))
    active = [e for e in entries if mentioned.intersection(entry_tags(e))]
    warnings: list[str] = []
    by_kind = {k: [e for e in active if e["kind"] == k] for k in KINDS}
    for kind, cap in CAPACITY.items():
        if len(by_kind[kind]) > cap:
            msg = (f"the prompt activates {len(by_kind[kind])} {kind}s; H3 Reference to Video takes {cap}"
                   + ("" if compliance == "strict" else f": the first {cap} in chain order were kept"))
            if compliance == "strict":
                raise TaggedRefError(msg + ". Remove some @tags or use compliance = soft.")
            warnings.append(msg)
            by_kind[kind] = by_kind[kind][:cap]
    aliases: dict[str, str] = {}
    presentation: list[dict] = []
    for i, e in enumerate(by_kind["picture"], 1):
        aliases[e["tag"]] = f"<Picture {i}>"
        presentation.append({"role": "picture", "tag": e["tag"], "label": f"<Picture {i}>", "entry": e})
    a = 0
    for i, e in enumerate(by_kind["video"], 1):
        if e.get("audio") is not None:
            a += 1
            aliases[e["audio_tag"]] = f"<Audio {a}>"
            presentation.append({"role": "audio", "tag": e["audio_tag"], "label": f"<Audio {a}>", "entry": e})
        aliases[e["tag"]] = f"<Video {i}>"
        presentation.append({"role": "video", "tag": e["tag"], "label": f"<Video {i}>", "entry": e})
    for e in by_kind["audio"]:
        a += 1
        aliases[e["tag"]] = f"<Audio {a}>"
        presentation.append({"role": "audio", "tag": e["tag"], "label": f"<Audio {a}>", "entry": e})
    return {"pictures": by_kind["picture"], "videos": by_kind["video"], "audios": by_kind["audio"],
            "aliases": aliases, "presentation": presentation, "warnings": warnings,
            "all_tags": {t for e in entries for t in entry_tags(e)}, "compliance": compliance}


def compile_prompt(registry: Any, prompt: Any, compliance: str = "strict") -> tuple[str, dict, str]:
    """(prompt with @tags replaced by native labels, bindings, report)."""
    text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    b = bindings(registry, text, compliance)
    warnings = b["warnings"]
    if compliance != "disabled":
        def replace(m):
            tag = m.group(1)
            if tag in b["aliases"]:
                return b["aliases"][tag]
            msg = (f"@{tag} is registered but was dropped (over H3's capacity)" if tag in b["all_tags"]
                   else f"@{tag} is not a registered reference")
            if compliance == "strict":
                raise TaggedRefError(msg + ". Register it with an H3 Tagged node, fix the spelling, or use "
                                           "compliance = soft (unknown @tags then pass through unchanged).")
            if msg not in warnings:
                warnings.append(msg)
            return m.group(0)
        text = ALIAS_RE.sub(replace, text)
    lines = [f"@{p['tag']} -> {p['label']}" for p in b["presentation"]] or ["no tagged reference is used"]
    if compliance == "disabled":
        lines.append("compliance disabled: @tags left in the prompt unchanged")
    lines += [f"warning: {w}" for w in warnings]
    return text, b, "\n".join(lines)


def ref2va_inputs(b: dict) -> dict:
    """The autogrow inputs of core's MiniMaxH3ReferenceToVideo for these bindings (index-paired soundtracks)."""
    out = {}
    if b["pictures"]:
        out["ref_images"] = {f"ref_image_{i}": e["value"] for i, e in enumerate(b["pictures"], 1)}
    if b["videos"]:
        out["ref_videos"] = {f"ref_video_{i}": e["value"] for i, e in enumerate(b["videos"], 1)}
        tracks = {f"ref_video_audio_{i}": e["audio"] for i, e in enumerate(b["videos"], 1) if e.get("audio") is not None}
        if tracks:
            out["ref_video_audios"] = tracks
    if b["audios"]:
        out["ref_audios"] = {f"ref_audio_{i}": e["value"] for i, e in enumerate(b["audios"], 1)}
    return out


def describe(registry: Any) -> str:
    entries = entries_of(registry)
    parts = []
    for e in entries:
        s = f"@{e['tag']} {e['kind']}"
        if e.get("audio_tag"):
            s += f" + @{e['audio_tag']} soundtrack"
        parts.append(s)
    return f"{len(entries)} tagged reference(s): " + ", ".join(parts) + f" ({make_registry(entries)['fingerprint'][:12]})"
