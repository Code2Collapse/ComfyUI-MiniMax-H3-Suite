# PORTED FROM: https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext@bfa7ac3 seam_correction.py
# (H3SeamCorrection.execute, Apache-2.0). The flow is upstream's; changed here: suite node id and category, the
# cut_detection switch that upstream's execute reads but never exposed, and a report output instead of console prints.
"""H3 Seam Correction - remove the exposure / colour steps (and optionally the motion jolts) where the segments of a
long H3 video meet, without touching real shot cuts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from comfy_api.latest import io

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mmx_utils import seam_correction as E  # noqa: E402


def correct_seams(images, fix_color_preset="medium", fix_motion_preset="off", fix_flash=False, flash_threshold=0.30,
                  blend_frames=2, use_gpu=True, cut_detection=True, cut_threshold=15.0):
    E.REPORT.clear()
    if images is None or images.shape[0] < 2:
        return images, "fewer than 2 frames: unchanged"
    photo = E._PHOTO_LEVELS.get(fix_color_preset)
    seam = E._SEAM_LEVELS.get(fix_motion_preset)
    flash_cfg = seam or E._SEAM_LEVELS["medium"]
    stat_window = int(photo["window"]) if photo else 3
    n = int(images.shape[0])
    device = E._work_device(images, use_gpu)
    E._say(f"device {device}, {n} frames, colour {fix_color_preset}, motion {fix_motion_preset}, flash "
           f"{'on' if fix_flash else 'off'}")
    min_gap = max(4, E._DETECT_WINDOW)

    def step_boundaries():
        stats = E._frame_stats(images, device)
        steps, _z = E._detect_step_boundaries(stats, E._DETECT_WINDOW, E._DETECT_Z, min_gap)
        b = sorted(set(s + 1 for s in steps if s + 1 < n))
        return list(b), list(b)

    if cut_detection:
        cuts = E._detect_shot_cuts(images, cut_threshold, device)
        if cuts is None:
            E._say("shot detection unavailable (PySceneDetect or OpenCV not installed): every exposure step is treated as "
                   "a seam, so a real shot cut is NOT told apart and may be corrected too (pip install scenedetect "
                   "opencv-python)")
            seams, edit_points = step_boundaries()
        else:
            seams, edit_points = [], sorted(cuts)
            for c in cuts:
                if c < 3 or c >= n - 3:
                    continue
                corr = E._spatial_similarity(images, left_start=max(0, c - 3), left_end=c, right_start=c,
                                             right_end=min(n, c + 3), device=device)
                if corr > 0.60:
                    seams.append(c)
                    E._say(f"boundary {c}: same shot, colour difference (correlation {corr:.3f}) -> corrected")
                else:
                    E._say(f"boundary {c}: a real shot cut (correlation {corr:.3f}) -> left alone")
    else:
        seams, edit_points = step_boundaries()
    if not seams:
        E._say("no seam found")
    if photo is None and seam is None and not fix_flash:
        E._say("all three corrections are off: unchanged")
        return images, "\n".join(E.REPORT)
    out = images.clone()
    if photo is not None and photo["mode"] == "flatten":
        before = E._luma_series(out, 0, n, device)
        _level, gain, clipped, _ref = E._correct_flatten(out, seams, float(photo["strength"]), device)
        after = E._luma_series(out, 0, n, device)
        E._say(f"colour max: per-frame level normalised; luma range {float(before.max() - before.min()):.4f} -> "
               f"{float(after.max() - after.min()):.4f} (gain {float(gain.min()):.2f}-{float(gain.max()):.2f})")
        if clipped:
            E._say(f"warning: {clipped}/{n} frames needed more than {E._GAIN_MAX}x gain and were only partly "
                   "brightened - they are crushed to near black; fix it at generation (re-run the segment, more "
                   "context, or an exposure reference)")
    elif photo is not None and seams:
        before = {int(b): E._boundary_step(out, b, stat_window, device) for b in seams}
        logs = E._correct_exposure(out, seams, edit_points, photo["mode"], photo["method"], float(photo["strength"]),
                                   stat_window, device, int(blend_frames))
        for bnd, s, e, a, b, warn in logs or []:
            diag, _ = E._describe_transform(a, b)
            after = E._boundary_step(out, bnd, stat_window, device)
            E._say(f"colour at frame {bnd} -> segment [{s},{e}): gain ({diag[0]:.3f},{diag[1]:.3f},{diag[2]:.3f}), "
                   f"luma step {before.get(int(bnd), 0.0):+.4f} -> {after:+.4f}" + (f" ({warn})" if warn else ""))
    if fix_flash:
        transients = E._detect_transient_frames(E._frame_stats(out, device), window=1, z_thresh=1.2, min_gap=min_gap)
        lw = torch.tensor(E._LUMA_W, device=out.device)
        area = max(0.0, min(1.0, float(flash_threshold)))
        for f in transients or []:
            if not 0 < f < out.shape[0] - 1:
                continue
            cur, prv, nxt = (out[i, ..., :3].float() for i in (f, f - 1, f + 1))
            luma = lambda x: torch.sum(x * lw, dim=-1, keepdim=True)  # noqa: E731
            target = (luma(prv) + luma(nxt)) / 2.0
            mask = (((luma(cur) - target).abs() / (target + 1e-6)) > 0.10).float()
            ratio = float(mask.mean())
            if ratio < area:
                continue
            out[f, ..., :3] = (cur * (1 - mask) + (prv + nxt) / 2.0 * mask).clamp(0, 1).to(out.dtype)
            E._say(f"flash frame {f} repaired ({ratio:.1%} of the pixels)")
    if seam is not None or fix_flash:
        for b in seams:
            kind, delta = E._classify_boundary(out, b, E._LOCAL_THRESHOLD, device)
            if delta < E._LOCAL_THRESHOLD or (kind == "flash" and not fix_flash) or (kind != "flash" and seam is None):
                continue
            cfg = flash_cfg if kind == "flash" else seam
            out = E._warp_blend_seam(out, b, window=cfg["window"], flow_strength=cfg["flow"],
                                     blend_strength=cfg["blend"], flow_iterations=cfg["iters"],
                                     flow_pyramid=cfg["pyramid"], device=device)
            E._say(f"frame {b}: {kind} seam warped and blended (step {delta:.3f})")
    if out.device != images.device or out.dtype != images.dtype:
        out = out.to(images.device).to(images.dtype)
    return out, "\n".join(E.REPORT)


LEVELS = list(E._LEVEL_ORDER)


class MiniMaxH3_SeamCorrection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3_SeamCorrection",
            display_name="H3 Seam Correction",
            category="MiniMax H3/Long",
            description=("Find the seams of a long video made of H3 segments and remove their exposure / colour steps "
                         "(affine or MKL colour transfer per segment), optionally the motion jolts (optical-flow warp "
                         "+ blend at the seam) and single flash frames. Real shot cuts are detected and left alone. "
                         "Works on the decoded frames of the whole film."),
            inputs=[
                io.Image.Input("images", tooltip="The decoded frames of the whole film."),
                io.Combo.Input("fix_color_preset", options=LEVELS, default="medium",
                               tooltip="Exposure / colour at the seams. low / medium: affine, high: MKL, max: flatten "
                                       "the level over the whole clip."),
                io.Combo.Input("fix_motion_preset", options=LEVELS, default="off",
                               tooltip="Optical-flow alignment + local blend at each seam, for motion jolts."),
                io.Boolean.Input("fix_flash", default=False,
                                 tooltip="Repair single bright / dark flash frames. Off if the shot has real flashes "
                                         "(lightning, explosions): those would be flattened."),
                io.Float.Input("flash_threshold", default=0.30, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of a frame that must flash before it is repaired (0.2-0.4)."),
                io.Boolean.Input("cut_detection", default=True,
                                 tooltip="Find real shot cuts (PySceneDetect if installed, else exposure steps) and "
                                         "leave them alone. Off: every detected step is treated as a seam."),
                io.Float.Input("cut_threshold", default=15.0, min=5.0, max=50.0, step=0.5,
                               tooltip="Shot-cut sensitivity: lower finds more cuts (10-20)."),
                io.Int.Input("blend_frames", default=2, min=0, max=8,
                             tooltip="Frames over which the level ramps at a seam."),
                io.Boolean.Input("use_gpu", default=True, tooltip="Statistics, colour and flow on the GPU if there is one."),
            ],
            outputs=[io.Image.Output(display_name="images"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, images, fix_color_preset="medium", fix_motion_preset="off", fix_flash=False, flash_threshold=0.30,
                cut_detection=True, cut_threshold=15.0, blend_frames=2, use_gpu=True) -> io.NodeOutput:
        out, report = correct_seams(images, fix_color_preset, fix_motion_preset, fix_flash, flash_threshold,
                                    blend_frames, use_gpu, cut_detection, cut_threshold)
        return io.NodeOutput(out, report, ui={"text": [report]})


__all__ = ["MiniMaxH3_SeamCorrection", "correct_seams"]
