"""Head Size Mode (owner 2026-10-13, C2C D8.21) - mmx_utils/head_room.py and its use in H3 Subject Crop.

Invariants from C2C docs/research/l820_minimax_character_replacement.md §7: the crop window contains the target head
(1); head_room_masks contain the target head and the original head (2); nothing connected -> bit-identical crops; the
head is scaled uniformly about the chin (never stretched)."""
import math

import numpy as np
import torch

from mmx_nodes.subject_crop import MiniMaxH3_SubjectCrop, MiniMaxH3_SubjectCropAdvanced
from mmx_utils import head_room as HR
from mmx_utils.pose_interop import pose_keypoint_to_wholebody

H, W, T = 240, 360, 5


def _vals(out):
    return out.result if hasattr(out, "result") else out


def _ellipse(cx, cy, rx, ry, h=H, w=W):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).float()


def _person(cx, chin_y, face_w, brow_chin, shoulder_w, canvas=(W, H)):
    """OpenPose JSON person (normalised 0..1, as ComfyUI's POSE_KEYPOINT): 18 body + 70 face points."""
    cw, ch = canvas
    body = np.zeros((18, 3), np.float32)
    body[2] = (cx - shoulder_w / 2, chin_y + 30, 1)                    # R shoulder
    body[5] = (cx + shoulder_w / 2, chin_y + 30, 1)                    # L shoulder
    body[0] = (cx, chin_y - brow_chin * 0.45, 1)                       # nose
    face = np.zeros((70, 3), np.float32)
    for i in range(17):                                                # jaw: a half ellipse, 0 and 16 at the ends
        a = math.pi * i / 16
        face[i] = (cx - face_w / 2 * math.cos(a), chin_y - brow_chin * 0.6 * (1 - math.sin(a)), 1)
    face[8] = (cx, chin_y, 1)                                          # chin
    for i in range(17, 27):                                            # brows
        face[i] = (cx - face_w * 0.35 + face_w * 0.7 * (i - 17) / 9, chin_y - brow_chin, 1)
    for arr in (body, face):
        arr[:, 0] /= cw
        arr[:, 1] /= ch
    return {"pose_keypoints_2d": body.reshape(-1).tolist(), "face_keypoints_2d": face.reshape(-1).tolist()}


def _pose(people, canvas=(W, H)):
    return [{"people": [p], "canvas_width": canvas[0], "canvas_height": canvas[1]} for p in people]


def _crop(node, images, masks, **head):
    if node is MiniMaxH3_SubjectCropAdvanced:
        mode = {"mode": "tracked", "crop_scale": 1.2, "aspect_ratio": 0.0}
        return _vals(node.execute(images, masks, mode, 0.1, 16, 0.0, **head))
    return _vals(node.execute(images, masks, {"mode": "combined", "crop_scale": 1.2, "aspect_ratio": 0.0},
                              16, 0.0, **head))


def _box(boxes, t=0):
    b = boxes[t][0]
    return b["x"], b["y"], b["width"], b["height"]


# ── the scale ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_scale_is_a_ratio_of_ratios_and_survives_resolution():
    orig, _ = pose_keypoint_to_wholebody(_pose([_person(180, 150, 60, 40, 120)] * T))
    # the reference: a bigger head for its shoulders (face/shoulder 0.65 vs 0.5) in a 4x larger picture
    ref, _ = pose_keypoint_to_wholebody(_pose([_person(720, 600, 260, 160, 400)], canvas=(1440, 960)))
    s, why = HR.head_scale_from_poses(orig, ref)
    assert math.isclose(s, 0.65 / 0.5, rel_tol=1e-3), why


def test_missing_points_fall_back_to_one_with_the_reason():
    orig, _ = pose_keypoint_to_wholebody(_pose([_person(180, 150, 60, 40, 120)] * T))
    headshot = _person(720, 600, 260, 160, 400)
    headshot["pose_keypoints_2d"] = [0.0] * 54                         # no shoulders in the photo
    ref, _ = pose_keypoint_to_wholebody(_pose([headshot], canvas=(1440, 960)))
    s, why = HR.head_scale_from_poses(orig, ref)
    assert s == 1.0 and "shoulders" in why
    assert HR.head_scale_from_poses(None, None)[0] == 1.0


def test_keep_original_composition_is_always_one_and_manual_scale_wins():
    assert HR.resolve_scale(HR.MODES[1], 1.7, None, None)[0] == 1.0
    assert HR.resolve_scale(HR.MODES[0], 1.4, None, None)[0] == 1.4
    assert HR.resolve_scale(HR.MODES[0], 9.0, None, None)[0] == HR.SCALE_MAX


def test_scaling_is_uniform_about_the_chin():
    m = _ellipse(180, 120, 30, 40)[None]
    heads, _ = HR.target_heads(m, None, 1.5)
    ys, xs = torch.nonzero(heads[0] > 0, as_tuple=True)
    oy, ox = torch.nonzero(m[0] > 0, as_tuple=True)
    assert math.isclose(int(ys.max()), int(oy.max()), abs_tol=1)              # the chin stays
    w0, h0 = int(ox.max() - ox.min() + 1), int(oy.max() - oy.min() + 1)
    w1, h1 = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    assert math.isclose(w1 / w0, h1 / h0, rel_tol=0.05) and math.isclose(h1 / h0, 1.5, rel_tol=0.05)


# ── the node ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_nothing_connected_gives_bit_identical_crops():
    images = torch.rand(T, H, W, 3)
    masks = torch.stack([_ellipse(180 + 4 * t, 120, 30, 38) for t in range(T)])
    for node in (MiniMaxH3_SubjectCrop, MiniMaxH3_SubjectCropAdvanced):
        a = _crop(node, images, masks)
        b = _crop(node, images, masks, head_size_mode=HR.MODES[0], head_scale=0.0)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
        assert len(a) == 5 and a[4].shape == a[1].shape


def test_window_and_head_room_contain_the_target_head():
    rx, ry, cx, cy = 30.0, 38.0, 180.0, 110.0
    images = torch.rand(T, H, W, 3)
    masks = torch.stack([_ellipse(cx, cy, rx, ry)] * T)
    crops, cmasks, boxes, report, room = _crop(MiniMaxH3_SubjectCrop, images, masks, head_scale=1.5)
    x, y, w, h = _box(boxes)
    target = _ellipse(cx, (cy + ry) - ry * 1.5, rx * 1.5, ry * 1.5)
    inside = target[y:y + h, x:x + w].sum() / target.sum()
    assert float(inside) > 0.999                                                 # invariant 1
    t_crop = torch.nn.functional.interpolate(target[y:y + h, x:x + w][None, None], size=room.shape[1:],
                                             mode="nearest")[0, 0]
    assert bool((room[0][t_crop > 0] > 0.5).all())                                # invariant 2: the new head
    o_crop = torch.nn.functional.interpolate(masks[0, y:y + h, x:x + w][None, None], size=room.shape[1:],
                                             mode="nearest")[0, 0]
    assert bool((room[0][o_crop > 0] > 0.5).all())                                # ... and the old one
    assert "head scale 1.500" in report


def test_poses_drive_the_room_and_keep_mode_does_not_grow_the_head():
    images = torch.rand(T, H, W, 3)
    masks = torch.stack([_ellipse(180, 125, 33, 42)] * T)
    orig = _pose([_person(180, 160, 66, 44, 132)] * T)                     # face/shoulder 0.5
    ref = _pose([_person(500, 400, 300, 180, 450)], canvas=(1000, 800))    # 0.667 -> scale 1.333
    match = _crop(MiniMaxH3_SubjectCrop, images, masks, original_pose=orig, reference_pose=ref)
    keep = _crop(MiniMaxH3_SubjectCrop, images, masks, head_size_mode=HR.MODES[1], original_pose=orig,
                 reference_pose=ref)
    assert "head scale 1.333" in match[3] and "keep original composition" in keep[3]
    mw, mh = _box(match[2])[2:]
    kw, kh = _box(keep[2])[2:]
    assert mw * mh > kw * kh                                                      # more room for the bigger head


def test_uncrop_with_head_room_keeps_the_exterior():
    from mmx_nodes.subject_crop import MiniMaxH3_SubjectUncrop
    images = torch.rand(T, H, W, 3)
    masks = torch.stack([_ellipse(180, 110, 30, 38)] * T)
    crops, _cm, boxes, _r, room = _crop(MiniMaxH3_SubjectCrop, images, masks, head_scale=1.3)
    out = _vals(MiniMaxH3_SubjectUncrop.execute(torch.zeros_like(crops), images, boxes, 0, room))[0]
    x, y, w, h = _box(boxes)
    outside = torch.ones(H, W, dtype=torch.bool)
    outside[y:y + h, x:x + w] = False
    assert torch.equal(out[:, outside], images[:, outside])
