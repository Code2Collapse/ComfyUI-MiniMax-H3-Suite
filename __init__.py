"""ComfyUI-MiniMaxSuite — H3 geometric spine + sampling guards (N1–N7, P1–P11)."""

from __future__ import annotations

from pathlib import Path

# ComfyUI imports its own top-level `utils` package and `nodes.py` before custom
# nodes load, so package dirs named `utils/`/`nodes/` here would shadow-collide in
# sys.modules and register ZERO nodes while looking healthy. They were renamed to
# mmx_utils/ and mmx_nodes/ once, in 2026-08.
#
# This used to RENAME those directories on every import. Mutating the filesystem as
# a side effect of an import is not acceptable in a plugin ComfyUI loads at startup:
# it fails on read-only installs, and it silently moves a directory someone may have
# created deliberately. Detect and refuse instead — the migration is long done, so
# reaching this means something is genuinely wrong.
_PKG = Path(__file__).resolve().parent
for _legacy in ("utils", "nodes"):
    if (_PKG / _legacy).is_dir():
        raise RuntimeError(
            f"ComfyUI-MiniMaxSuite: found a legacy '{_legacy}/' directory at {_PKG / _legacy}. "
            f"It collides with ComfyUI's own top-level '{_legacy}' and will stop this pack "
            f"from registering any nodes. Rename it to 'mmx_{_legacy}' and update imports."
        )

import logging
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

_LOG = logging.getLogger(__name__)
_LOAD_ERRORS: list[str] = []

_NODE_SPECS: tuple[tuple[str, str], ...] = (
    (".mmx_nodes.track_crop", "MiniMaxH3_TrackCrop"),
    (".mmx_nodes.mask_prep", "MiniMaxH3_MaskPrep"),
    (".mmx_nodes.stitch_back", "MiniMaxH3_StitchBack"),
    (".mmx_nodes.control_hints", "MiniMaxH3_ControlHints"),
    (".mmx_nodes.masked_replace", "MiniMaxH3_MaskedReplace"),
    (".mmx_nodes.detail_reinject", "MiniMaxH3_DetailReinject"),
    (".mmx_nodes.frame_handles", "MiniMaxH3_FrameHandles"),
    (".mmx_nodes.protected_layer_guard", "MiniMaxH3_ProtectedLayerGuard"),
    (".mmx_nodes.accelerator_conflict", "MiniMaxH3_AcceleratorConflict"),
    (".mmx_nodes.legal_scheduler", "MiniMaxH3_LegalScheduler"),
    (".mmx_nodes.sigma_shift_locked", "MiniMaxH3_SigmaShiftLocked"),
    (".mmx_nodes.per_frame_denoise", "MiniMaxH3_PerFrameDenoise"),
    (".mmx_nodes.block_cache", "MiniMaxH3_BlockCacheT8"),
    (".mmx_nodes.sigma_inspector", "MiniMaxH3_SigmaInspector"),
    (".mmx_nodes.turbo_lora", "MiniMaxH3_TurboLoRA"),
    (".mmx_nodes.turbo_sampler", "MiniMaxH3_TurboSampler"),
    (".mmx_nodes.audio_quality_gate", "MiniMaxH3_AudioQualityGate"),
    (".mmx_nodes.ocio_bridge", "MiniMaxH3_OCIOBridge"),
    (".mmx_nodes.color_roundtrip_qc", "MiniMaxH3_ColorRoundTripQC"),
    (".mmx_nodes.pose_hints", "MiniMaxH3_StrongestPose"),
    (".mmx_nodes.depth_hints", "MiniMaxH3_TemporalDepth"),
    (".mmx_nodes.drift_qc", "MiniMaxH3_DriftQC"),
    (".mmx_nodes.region_mask", "MiniMaxH3_RegionMask"),
    (".mmx_nodes.family_presets", "MiniMaxH3_FamilyPresets"),
    (".mmx_nodes.edit_validator", "MiniMaxH3_EditValidator"),
    (".mmx_nodes.dcc_bridge", "MiniMaxH3_DCCBridge"),
    (".mmx_nodes.puppeteer", "MiniMaxH3_PosePuppeteer"),
    (".mmx_nodes.hdr_roundtrip", "MiniMaxH3_HDRRoundtrip"),
)


def _load_nodes() -> list[type[io.ComfyNode]]:
    nodes: list[type[io.ComfyNode]] = []
    for mod_path, cls_name in _NODE_SPECS:
        try:
            if mod_path.endswith("track_crop"):
                from .mmx_nodes.track_crop import MiniMaxH3_TrackCrop

                nodes.append(MiniMaxH3_TrackCrop)
            elif mod_path.endswith("mask_prep"):
                from .mmx_nodes.mask_prep import MiniMaxH3_MaskPrep

                nodes.append(MiniMaxH3_MaskPrep)
            elif mod_path.endswith("stitch_back"):
                from .mmx_nodes.stitch_back import MiniMaxH3_StitchBack

                nodes.append(MiniMaxH3_StitchBack)
            elif mod_path.endswith("control_hints"):
                from .mmx_nodes.control_hints import MiniMaxH3_ControlHints

                nodes.append(MiniMaxH3_ControlHints)
            elif mod_path.endswith("masked_replace"):
                from .mmx_nodes.masked_replace import MiniMaxH3_MaskedReplace

                nodes.append(MiniMaxH3_MaskedReplace)
            elif mod_path.endswith("detail_reinject"):
                from .mmx_nodes.detail_reinject import MiniMaxH3_DetailReinject

                nodes.append(MiniMaxH3_DetailReinject)
            elif mod_path.endswith("frame_handles"):
                from .mmx_nodes.frame_handles import MiniMaxH3_FrameHandles

                nodes.append(MiniMaxH3_FrameHandles)
            elif mod_path.endswith("protected_layer_guard"):
                from .mmx_nodes.protected_layer_guard import MiniMaxH3_ProtectedLayerGuard

                nodes.append(MiniMaxH3_ProtectedLayerGuard)
            elif mod_path.endswith("accelerator_conflict"):
                from .mmx_nodes.accelerator_conflict import MiniMaxH3_AcceleratorConflict

                nodes.append(MiniMaxH3_AcceleratorConflict)
            elif mod_path.endswith("legal_scheduler"):
                from .mmx_nodes.legal_scheduler import MiniMaxH3_LegalScheduler

                nodes.append(MiniMaxH3_LegalScheduler)
            elif mod_path.endswith("sigma_shift_locked"):
                from .mmx_nodes.sigma_shift_locked import MiniMaxH3_SigmaShiftLocked

                nodes.append(MiniMaxH3_SigmaShiftLocked)
            elif mod_path.endswith("per_frame_denoise"):
                from .mmx_nodes.per_frame_denoise import MiniMaxH3_PerFrameDenoise

                nodes.append(MiniMaxH3_PerFrameDenoise)
            elif mod_path.endswith("block_cache"):
                from .mmx_nodes.block_cache import MiniMaxH3_BlockCacheT8

                nodes.append(MiniMaxH3_BlockCacheT8)
            elif mod_path.endswith("sigma_inspector"):
                from .mmx_nodes.sigma_inspector import MiniMaxH3_SigmaInspector

                nodes.append(MiniMaxH3_SigmaInspector)
            elif mod_path.endswith("turbo_lora"):
                from .mmx_nodes.turbo_lora import MiniMaxH3_TurboLoRA

                nodes.append(MiniMaxH3_TurboLoRA)
            elif mod_path.endswith("turbo_sampler"):
                from .mmx_nodes.turbo_sampler import MiniMaxH3_TurboSampler

                nodes.append(MiniMaxH3_TurboSampler)
            elif mod_path.endswith("audio_quality_gate"):
                from .mmx_nodes.audio_quality_gate import MiniMaxH3_AudioQualityGate

                nodes.append(MiniMaxH3_AudioQualityGate)
            elif mod_path.endswith("ocio_bridge"):
                from .mmx_nodes.ocio_bridge import MiniMaxH3_OCIOBridge

                nodes.append(MiniMaxH3_OCIOBridge)
            elif mod_path.endswith("color_roundtrip_qc"):
                from .mmx_nodes.color_roundtrip_qc import MiniMaxH3_ColorRoundTripQC

                nodes.append(MiniMaxH3_ColorRoundTripQC)
            elif mod_path.endswith("pose_hints"):
                from .mmx_nodes.pose_hints import MiniMaxH3_StrongestPose

                nodes.append(MiniMaxH3_StrongestPose)
            elif mod_path.endswith("depth_hints"):
                from .mmx_nodes.depth_hints import MiniMaxH3_TemporalDepth

                nodes.append(MiniMaxH3_TemporalDepth)
            elif mod_path.endswith("drift_qc"):
                from .mmx_nodes.drift_qc import MiniMaxH3_DriftQC

                nodes.append(MiniMaxH3_DriftQC)
            elif mod_path.endswith("region_mask"):
                from .mmx_nodes.region_mask import MiniMaxH3_RegionMask

                nodes.append(MiniMaxH3_RegionMask)
            elif mod_path.endswith("family_presets"):
                from .mmx_nodes.family_presets import MiniMaxH3_FamilyPresets

                nodes.append(MiniMaxH3_FamilyPresets)
            elif mod_path.endswith("edit_validator"):
                from .mmx_nodes.edit_validator import MiniMaxH3_EditValidator

                nodes.append(MiniMaxH3_EditValidator)
            elif mod_path.endswith("dcc_bridge"):
                from .mmx_nodes.dcc_bridge import MiniMaxH3_DCCBridge

                nodes.append(MiniMaxH3_DCCBridge)
            elif mod_path.endswith("puppeteer"):
                from .mmx_nodes.puppeteer import MiniMaxH3_PosePuppeteer

                nodes.append(MiniMaxH3_PosePuppeteer)
            elif mod_path.endswith("hdr_roundtrip"):
                from .mmx_nodes.hdr_roundtrip import MiniMaxH3_HDRRoundtrip

                nodes.append(MiniMaxH3_HDRRoundtrip)
        except Exception as exc:
            msg = f"ComfyUI-MiniMaxSuite: failed to import {cls_name}: {exc}"
            _LOAD_ERRORS.append(msg)
            _LOG.error(msg)
    return nodes


class MiniMaxH3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _load_nodes()


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()


WEB_DIRECTORY = "web"

__all__ = ["comfy_entrypoint", "MiniMaxH3Extension", "WEB_DIRECTORY"]
