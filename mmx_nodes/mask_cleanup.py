# PORTED FROM: MaskVidExperiments (third_party/MaskVidExperiments/nodes_subject_crop.py) by drozbay
# Licence: GPL-3.0 — direct copy authorised by owner; attribution retained.
"""Remove specks and flicker from video mask batches before crop planning."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from comfy_api.latest import io

CATEGORY = "MiniMax H3/Mask"


def _clean_mask_stack(masks, value_thresh, min_pixels, min_frames):
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary, 0

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary, 0

    span = np.zeros(count + 1, dtype=np.int64)
    for label, location in enumerate(ndimage.find_objects(labels), start=1):
        if location is not None:
            span[label] = location[0].stop - location[0].start

    peak = np.zeros(count + 1, dtype=np.int64)
    for frame in labels:
        peak = np.maximum(peak, np.bincount(frame.ravel(), minlength=count + 1))

    span[0] = 0
    peak[0] = 0
    subject = peak >= 0.5 * peak.max() if peak.max() > 0 else np.zeros_like(peak, dtype=bool)
    keep = (peak >= min_pixels) & (subject | (span >= min_frames))
    keep[0] = False
    kept = int(keep[1:].sum())
    return keep[labels], count - kept


def _open_reconstruct(masks, value_thresh, radius, min_frames):
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary, 0

    spatial = np.zeros((3, 3, 3), dtype=bool)
    spatial[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    eroded = ndimage.binary_erosion(binary, structure=spatial, iterations=radius, border_value=1)

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary, 0

    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[eroded])] = True
    if min_frames > 1:
        for label, location in enumerate(ndimage.find_objects(labels), start=1):
            if location is not None and location[0].stop - location[0].start < min_frames:
                keep[label] = False
    keep[0] = False
    kept = int(keep[1:].sum())
    return keep[labels], count - kept


class MiniMaxH3_MaskCleanup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3_MaskCleanup",
            display_name="H3 Mask Cleanup",
            category=CATEGORY,
            description="Removes specks and brief flickering blobs from a video mask batch while keeping the real subject, including its soft edges.",
            inputs=[
                io.Mask.Input("masks", tooltip="Mask batch to clean, one mask per frame."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Mask values above this count as subject when finding blobs."),
                io.DynamicCombo.Input(
                    "method",
                    tooltip="shrink_grow: shrinks the mask so thin specks vanish, then restores the surviving blobs' exact shapes. components: drops blobs that are both small and short-lived.",
                    options=[
                        io.DynamicCombo.Option("shrink_grow", [
                            io.Int.Input("shrink", default=4, min=1,
                                         tooltip="Shrink distance in pixels. Blobs thinner than about twice this are removed."),
                            io.Int.Input("min_frames", default=1, min=1,
                                         tooltip="Surviving blobs must also persist at least this many frames. 1 disables the temporal check."),
                        ]),
                        io.DynamicCombo.Option("components", [
                            io.Int.Input("min_pixels", default=32, min=1,
                                         tooltip="Blobs whose largest single-frame area is below this many pixels are removed."),
                            io.Int.Input("min_frames", default=2, min=1,
                                         tooltip="Small blobs must persist at least this many frames to survive."),
                        ]),
                    ],
                ),
                io.Int.Input("edge_grow", default=4, min=0,
                             tooltip="Grow kept regions by this many pixels so the subject's soft edges are preserved."),
            ],
            outputs=[
                io.Mask.Output(display_name="masks", tooltip="Cleaned masks, same size and soft values as the input."),
                io.String.Output(display_name="report", tooltip="How many blobs were removed and which method ran."),
            ],
        )

    @classmethod
    def execute(cls, masks, threshold, method, edge_grow) -> io.NodeOutput:
        n_frames = masks.shape[0]
        if method["method"] == "shrink_grow":
            keep, removed = _open_reconstruct(masks, threshold, method["shrink"], method["min_frames"])
            mode = f"shrink_grow (radius={method['shrink']}, min_frames={method['min_frames']})"
        else:
            keep, removed = _clean_mask_stack(masks, threshold, method["min_pixels"], method["min_frames"])
            mode = (f"components (min_pixels={method['min_pixels']}, "
                    f"min_frames={method['min_frames']})")
        if edge_grow > 0:
            spatial = np.zeros((3, 3, 3), dtype=bool)
            spatial[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            keep = ndimage.binary_dilation(keep, structure=spatial, iterations=edge_grow)
        gate = torch.from_numpy(keep).to(dtype=masks.dtype, device=masks.device)
        report = (f"method: {mode}\nframes: {n_frames}\n"
                  f"blobs_removed: {removed}\nedge_grow: {edge_grow}px")
        return io.NodeOutput(masks * gate, report)
