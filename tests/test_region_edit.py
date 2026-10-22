"""H3 Region Edit (owner A1, L8.18): one node = H3 Source AV Latent + H3 Mask-Aware ControlNet, with the ControlNet
kept out of the plate the latent holds."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("comfy.nested_tensor")

import mmx_nodes.mask_aware_control as mac  # noqa: E402
from mmx_nodes import region_edit as RE  # noqa: E402
from tests.test_source_av_latent import FakeH3VAE  # noqa: E402


class _FakeModel:
    def clone(self):
        return self


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class _FakePatch:
        def __init__(self, control_net, vae, control_video, mask, source_video, strength, sigma_start, sigma_end,
                     **kw):
            seen.update(kw, control_video=control_video, mask=mask, source_video=source_video, strength=strength)

        def register(self, _model):
            seen["registered"] = True

    monkeypatch.setattr(mac, "MaskAwareControlPatch", _FakePatch)
    monkeypatch.setattr(mac, "_require_core", lambda: None)
    return seen


def plate(n=22, h=64, w=96):
    return torch.rand(n, h, w, 3)


def half_mask(n=22, h=64, w=96):
    m = torch.zeros(n, h, w)
    m[:, :, : w // 2] = 1.0
    return m


def run(**kw):
    args = dict(model=_FakeModel(), video_vae=FakeH3VAE(), images=plate(), mask=half_mask())
    args.update(kw)
    out = RE.MiniMaxH3_RegionEdit.execute(**args)
    return out.result if hasattr(out, "result") else out.args


def test_controlnet_stays_out_of_the_plate(captured):
    m, latent, report = run(control_net=object(), control_video=plate())
    assert captured["registered"] and captured["preserved_strength"] == 0.0
    assert captured["inpaint"] is False and captured["control_video"] is not None and captured["mask"] is None
    assert "noise_mask" in latent and "outside the mask the latent keeps the plate" in report


def test_everywhere_is_cores_behaviour(captured):
    run(control_net=object(), control_video=plate(), controlnet_where="everywhere")
    assert captured["preserved_strength"] == 1.0


def test_auto_hint_inpaints_from_the_plate_without_a_control_video(captured):
    images = plate()
    run(control_net=object(), images=images)
    assert captured["inpaint"] is True and captured["source_video"] is images and captured["mask"] is not None
    assert captured["control_video"] is None


def test_both_and_refusals(captured):
    run(control_net=object(), control_video=plate(), hint="both")
    assert captured["inpaint"] is True and captured["control_video"] is not None
    with pytest.raises(ValueError, match="needs a control_video"):
        run(control_net=object(), hint="control video")
    with pytest.raises(ValueError, match="needs the mask"):
        run(control_net=object(), mask=None)


def test_without_a_controlnet_it_is_a_masked_edit(captured):
    model = _FakeModel()
    m, latent, report = run(model=model)
    assert m is model and "registered" not in captured and "masked edit only" in report


def test_latent_is_the_source_av_latent():
    from mmx_nodes.source_av_latent import build_source_av_latent
    images, mask = plate(), half_mask()
    _m, latent, _r = run(images=images, mask=mask)
    ref, _ = build_source_av_latent(images, FakeH3VAE(), None, None, mask, "keep", RE.SIZE_POLICIES[0])
    for a, b in zip(latent["noise_mask"].unbind(), ref["noise_mask"].unbind()):
        assert torch.equal(a, b)
