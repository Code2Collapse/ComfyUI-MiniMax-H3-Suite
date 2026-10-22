"""H3 Latent Upscale (Model) (ported from bbaudio-2025/Comfyui-MMH3-UltimateUpscale + LBH-123-AI, MIT): architecture
detection and strict load from safetensors, pickle refused, the 24-channel check, chunk cross-fade, audio untouched."""
from __future__ import annotations

import pytest
import torch

nt = pytest.importorskip("comfy.nested_tensor")
st = pytest.importorskip("safetensors.torch")

from mmx_nodes import latent_upscale_model as N  # noqa: E402
from mmx_utils import h3_latent_upscaler as U  # noqa: E402


def tiny(in_channels=24, attn=False, seed=0):
    torch.manual_seed(seed)
    m = U.LatentResizer3D(in_channels=in_channels, in_blocks=2, out_blocks=3, channels=32, dropout=0.0, attn=attn,
                          temporal_every=2, temporal_kernel=3)
    with torch.no_grad():
        for p in m.parameters():          # the zero-initialised branches would hide a wrong load
            p.add_(torch.randn_like(p) * 0.02)
    return m.eval()


def test_architecture_detected_and_loaded_strictly(tmp_path):
    m = tiny()
    path = tmp_path / "tiny.safetensors"
    st.save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(path))
    sd = U.load_state_dict(str(path))
    cfg = U.detect_arch(sd)
    assert (cfg["in_blocks"], cfg["out_blocks"], cfg["channels"], cfg["temporal_kernel"], cfg["attn"]) == (2, 3, 32, 3,
                                                                                                         False)
    loaded = U.build_model(sd, "tiny")
    x = torch.randn(1, 24, 3, 4, 6)
    with torch.no_grad():
        assert torch.allclose(loaded(x, scale=2.0, target_size=(3, 8, 12)), m(x, scale=2.0, target_size=(3, 8, 12)))


def test_prefixed_and_attention_checkpoints(tmp_path):
    m = tiny(attn=True)
    path = tmp_path / "prefixed.safetensors"
    st.save_file({"upscaler." + k: v.contiguous() for k, v in m.state_dict().items()}, str(path))
    sd = U.load_state_dict(str(path))
    assert U.detect_arch(sd)["attn"] is True
    U.build_model(sd)


def test_pickle_refused_before_reading(tmp_path):
    with pytest.raises(U.UpscalerError, match="Pickle"):
        U.load_state_dict(str(tmp_path / "does_not_exist.pth"))


def test_wrong_network_refused():
    with pytest.raises(U.UpscalerError, match="24"):
        U.build_model(tiny(in_channels=16).state_dict(), "other")
    sd = tiny().state_dict()
    sd.pop("conv_out.bias")
    with pytest.raises(U.UpscalerError, match="does not match"):
        U.build_model(sd, "broken")


@pytest.mark.parametrize("t,chunk,ov", [(10, 0, 4), (10, 4, 1), (37, 8, 4), (37, 36, 4), (5, 5, 2), (37, 7, 9)])
def test_chunk_ranges_cover_every_step(t, chunk, ov):
    r = U.chunk_ranges(t, chunk, ov)
    assert r[0][0] == 0 and r[-1][1] == t
    for (s0, e0), (s1, e1) in zip(r, r[1:]):
        assert s0 < s1 < e0 <= e1


class FrameLocal(torch.nn.Module):
    """T-local stand-in: chunked and whole must agree exactly."""

    def forward(self, x, scale=None, target_size=None):
        b, c, t, _h, _w = x.shape
        return torch.nn.functional.interpolate(x, size=target_size, mode="trilinear", align_corners=False) + 0.5


def test_chunked_cross_fade_is_exact_for_a_frame_local_model():
    v = torch.randn(1, 24, 13, 4, 6)
    whole = U.run_model(FrameLocal(), v, 8, 12, "cpu", torch.float32, 0)
    chunked = U.run_model(FrameLocal(), v, 8, 12, "cpu", torch.float32, 5, 2)
    assert whole.shape == (1, 24, 13, 8, 12) and torch.allclose(whole, chunked, atol=1e-6)


def av(t=4, h=4, w=6, ticks=10, mask=False):
    lat = {"samples": nt.NestedTensor((torch.randn(1, 24, t, h, w), torch.randn(1, 32, 2, ticks)))}
    if mask:
        lat["noise_mask"] = nt.NestedTensor((torch.ones(1, 1, t, h, w).index_fill_(3, torch.tensor([0]), 0.0),
                                             torch.ones(1, 1, 2, ticks)))
    return lat


def test_interpolation_keeps_audio_and_resizes_the_mask():
    lat = av(mask=True)
    out, report = N.upscale(lat, "bilinear", U.NO_MODEL, 192, 128)
    v, a = out["samples"].unbind()
    v0, a0 = lat["samples"].unbind()
    assert v.shape == (1, 24, 4, 8, 12) and torch.equal(a, a0)
    vm, am = out["noise_mask"].unbind()
    assert vm.shape == (1, 1, 4, 8, 12) and float(vm[0, 0, 0, :2].max()) == 0.0 and float(vm[0, 0, 0, 2:].min()) == 1
    assert "audio untouched" in report and "bilinear" in report


def test_model_path_with_a_loaded_network(monkeypatch):
    m = tiny()
    monkeypatch.setattr(N, "load", lambda name, precision, device: m)
    lat = av()
    out, report = N.upscale(lat, "model", "tiny.safetensors", 192, 128, device="cpu")
    v, a = out["samples"].unbind()
    assert v.shape == (1, 24, 4, 8, 12) and torch.equal(a, lat["samples"].unbind()[1])
    assert "whole clip" in report
    out2, report2 = N.upscale(lat, "model", "tiny.safetensors", 192, 128, device="cpu", chunk_latent_steps=2,
                              chunk_overlap_steps=1)
    assert out2["samples"].unbind()[0].shape == v.shape and "approximate" in report2


def test_refusals_and_noop():
    lat = av()
    with pytest.raises(U.UpscalerError, match="only enlarges"):
        N.upscale(lat, "model", "x.safetensors", 64, 64, device="cpu")
    with pytest.raises(U.UpscalerError, match="huggingface"):
        N.upscale(lat, "model", U.NO_MODEL, 192, 128, device="cpu")
    same, report = N.upscale(lat, "bilinear", U.NO_MODEL, 96, 64)
    assert same["samples"] is lat["samples"] and "unchanged" in report
    bad = {"samples": nt.NestedTensor((torch.randn(1, 16, 4, 4, 6), torch.randn(1, 32, 2, 10)))}
    with pytest.raises(U.UpscalerError, match="24-channel"):
        N.upscale(bad, "bilinear", U.NO_MODEL, 192, 128)
    with pytest.raises(ValueError, match="single-stream"):
        N.upscale({"samples": torch.zeros(1, 24, 4, 4, 6)}, "bilinear", U.NO_MODEL, 192, 128)


def test_model_list_hides_pickles():
    assert U.list_models(["a.pth", "b.safetensors", "c.ckpt", "d.SFT"]) == ["b.safetensors", "d.SFT"]
    assert U.list_models(["a.pth"]) == [U.NO_MODEL]
