"""L8.20 geometry invariants of the subject crop chain the owner's character-replacement workflow runs
(MVEx_SubjectCrop / Uncrop, ported as MiniMaxH3_SubjectCrop / SubjectUncrop; token snap of MaskToLatentSpace).

Invariants (docs/research/l820_minimax_character_replacement.md §7): crop -> uncrop maps a point back within 1 px
and keeps the exterior exactly; every masked pixel lands in a regenerated latent cell. And the measured limitation
behind "the new head comes out too small": a tight crop leaves no room for a bigger head (§3).
"""
import math

import torch
import torch.nn.functional as F

from mmx_nodes.mask_to_latent import _token_snap
from mmx_nodes.subject_crop import MiniMaxH3_SubjectCrop, MiniMaxH3_SubjectUncrop

H, W, T = 240, 360, 5


def _vals(out):
    return out.result if hasattr(out, "result") else out


def _ellipse(cx, cy, rx, ry, h=H, w=W):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).float()


def _gradient_plate():
    # band-limited content: a crop -> resize -> inverse resize round trip is exact only for content below Nyquist
    yy = torch.linspace(0, 1, H).view(1, H, 1, 1)
    xx = torch.linspace(0, 1, W).view(1, 1, W, 1)
    return torch.cat([xx.expand(T, H, W, 1), yy.expand(T, H, W, 1), (0.5 * xx + 0.5 * yy).expand(T, H, W, 1)], -1)


def _crop(images, masks, scale=1.2, div=32, mp=0.0):
    return _vals(MiniMaxH3_SubjectCrop.execute(images, masks, {"mode": "combined", "crop_scale": scale,
                                                               "aspect_ratio": 0.0}, div, mp))


def test_point_maps_into_the_crop_and_back_within_one_pixel():
    masks = torch.stack([_ellipse(180, 120, 40, 50)] * T)
    images = torch.zeros(T, H, W, 3)
    px, py = 196, 101                                                     # a marked point on the subject
    images[:, py, px] = 1.0
    crops, cmasks, boxes, _ = _crop(images, masks, mp=0.25)
    b = boxes[0][0]
    sx, sy = crops.shape[2] / b["width"], crops.shape[1] / b["height"]
    peak = crops[0, ..., 0]
    yy, xx = divmod(int(torch.argmax(peak)), peak.shape[1])
    assert abs(xx - (px - b["x"] + 0.5) * sx + 0.5) <= 1.0 + sx / 2 and abs(yy - (py - b["y"] + 0.5) * sy + 0.5) <= 1.0 + sy / 2


def test_identity_round_trip_keeps_the_exterior_exactly_and_the_interior_close():
    images = _gradient_plate()
    masks = torch.stack([_ellipse(180, 120, 40, 50)] * T)
    crops, cmasks, boxes, _ = _crop(images, masks, mp=0.0)               # no resize: exact to float rounding
    out = _vals(MiniMaxH3_SubjectUncrop.execute(crops, images, boxes, 16))[0]
    b0 = boxes[0][0]
    box0 = torch.zeros(H, W, dtype=torch.bool)
    box0[b0["y"]:b0["y"] + b0["height"], b0["x"]:b0["x"] + b0["width"]] = True
    assert torch.equal(out[:, ~box0], images[:, ~box0])                  # outside the crop: untouched
    assert float((out - images).abs().max()) < 1e-6                      # inside: the feather blend's rounding only
    crops2, _m, boxes2, _ = _crop(images, masks, mp=0.25)                # resized crop: close inside, exact outside
    out2 = _vals(MiniMaxH3_SubjectUncrop.execute(crops2, images, boxes2, 16))[0]
    b = boxes2[0][0]
    inside = torch.zeros(H, W, dtype=torch.bool)
    inside[b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]] = True
    assert torch.equal(out2[:, ~inside], images[:, ~inside])
    assert float((out2[:, inside] - images[:, inside]).abs().max()) < 0.02


def test_every_masked_pixel_lands_in_a_regenerated_latent_cell():
    g = torch.Generator().manual_seed(1)
    m = (torch.rand(3, 100, 130, generator=g) > 0.995).float() * torch.rand(3, 100, 130, generator=g)
    snapped = _token_snap(m, 32, "max")
    assert bool((snapped >= m).all())


def test_tight_crop_leaves_no_room_for_a_larger_head():
    # the limitation measured in docs/evidence/L8.20/headroom.json, pinned at fixture size: crop_scale 1.2 around
    # the original head; a head 1.5x larger about the same neck point does not fit in the crop window
    rx, ry, cx, cy = 30.0, 38.0, 180.0, 110.0
    masks = torch.stack([_ellipse(cx, cy, rx, ry)] * T)
    _crops, _cm, boxes, _ = _crop(_gradient_plate(), masks, scale=1.2, div=16)
    b = boxes[0][0]
    neck = cy + ry
    shares = {}
    for r in (1.0, 1.5):
        tgt = _ellipse(cx, neck - ry * r, rx * r, ry * r)
        shares[r] = float(tgt[b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]].sum()) / float(tgt.sum())
    assert math.isclose(shares[1.0], 1.0)
    assert shares[1.5] < 0.95
