"""CPU tests for N1 Latent Colour Anchor — deterministic, no real model weights."""

from __future__ import annotations

import copy
import logging

import pytest
import torch

from mmx_utils.latent_color_anchor import (
    anchor_video,
    build_preserved_packed,
    make_hook,
    split_video,
)

COMFY_BACKEND = "stub"
try:
    import comfy.sampler_helpers
    import comfy.utils

    COMFY_BACKEND = "real"
except ImportError:
    comfy = None  # type: ignore[assignment]


class _FakeNested:
    is_nested = True

    def __init__(self, members):
        self._members = tuple(members)

    def unbind(self):
        return self._members


class _FlowSampling:
    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        return 1.0 - float(percent)


class _FakeModel:
    def __init__(
        self,
        latent_shapes,
        *,
        with_sampling: bool = True,
        audio_scale: float = 2.0,
    ):
        self.latent_shapes = list(latent_shapes)
        self.model_sampling = _FlowSampling() if with_sampling else None
        self._audio_scale = audio_scale

    def process_latent_in(self, x):
        if getattr(x, "is_nested", False):
            video, audio = x.unbind()
            return _FakeNested((video, audio * self._audio_scale))
        if isinstance(x, torch.Tensor) and x.ndim == 5:
            return x
        tensors = comfy.utils.unpack_latents(x, self.latent_shapes)
        video, audio = tensors[0], tensors[1]
        repacked, _ = comfy.utils.pack_latents([
            video,
            audio * self._audio_scale,
        ])
        return repacked


def _prepare_mask(noise_mask, shape, device):
    if COMFY_BACKEND == "real":
        return comfy.sampler_helpers.prepare_mask(noise_mask, shape, device)
    # Minimal stub: broadcast [T,h,w] or [B,C,T,h,w] to shape.
    m = noise_mask.to(device)
    if m.shape == shape:
        return m
    if m.ndim == 3 and len(shape) == 5:
        return m.unsqueeze(0).unsqueeze(0).expand(shape)
    return m.expand(shape)


def _pack(latents):
    if COMFY_BACKEND == "real":
        return comfy.utils.pack_latents(latents)
    shapes = [t.shape for t in latents]
    flat = [t.reshape(t.shape[0], 1, -1) for t in latents]
    return torch.cat(flat, dim=-1), shapes


def _unpack(packed, shapes):
    if COMFY_BACKEND == "real":
        return comfy.utils.unpack_latents(packed, shapes)
    out = []
    cursor = 0
    for shape in shapes:
        n = 1
        for d in shape[1:]:
            n *= d
        chunk = packed[:, :, cursor:cursor + n]
        out.append(chunk.reshape(shape))
        cursor += n
    return out


def _video_drift(source: torch.Tensor) -> torch.Tensor:
    """Per-frame per-channel constant offset growing with t (simulates #55 drift)."""
    b, c, t, h, w = source.shape
    drift = torch.zeros_like(source)
    for ti in range(t):
        for ch in range(c):
            drift[:, ch, ti, :, :] = (ti + 1) * 0.02 * (ch + 1)
    return source + drift


def _center_regenerate_mask(shape, inner=2):
    """Denoise mask: 1 = regenerate (centre), 0 = preserve (outside)."""
    b, c, t, h, w = shape
    m = torch.zeros(b, c, t, h, w)
    h0, h1 = h // 2 - inner, h // 2 + inner
    w0, w1 = w // 2 - inner, w // 2 + inner
    m[:, :, :, h0:h1, w0:w1] = 1.0
    return m


def _single(video: torch.Tensor) -> torch.Tensor:
    """The layout ComfyUI hands a post-CFG hook for a NON-nested latent: the raw
    [B, C, T, h, w] tensor, unpacked.

    CFGGuider.sample only calls pack_latents when the latent is nested; a
    single-stream latent is never flattened. An earlier version of these tests
    packed single-stream latents into [B, 1, N] - a layout the sampler never
    produces - and then indexed the result as 5-D, which is where four of its
    failures came from. The library handles a flat input too; the tests now
    describe what actually happens.
    """
    return video.clone()


def _run_hook(
    source,
    denoised,
    noise_mask,
    shapes,
    sigma: float = 0.5,
    **hook_kw,
):
    hook = make_hook(
        source,
        noise_mask,
        hook_kw.get("strength", 1.0),
        hook_kw.get("mode", "mean"),
        hook_kw.get("per_frame", True),
        hook_kw.get("min_preserved", 0.05),
        hook_kw.get("start_percent", 0.0),
        hook_kw.get("end_percent", 0.85),
        _prepare_mask,
        _pack,
        _unpack,
        shapes_hint=shapes,
        logger=logging.getLogger("test_latent_color_anchor"),
    )
    model = _FakeModel(shapes, with_sampling=hook_kw.get("with_sampling", True))
    args = {
        "denoised": denoised.clone(),
        "sigma": torch.tensor([sigma]),
        "model": model,
    }
    return hook(args), args["denoised"]


@pytest.fixture(scope="module", autouse=True)
def _report_comfy_backend():
    if COMFY_BACKEND != "real":
        pytest.skip("ComfyUI pack/unpack not importable — latent colour anchor tests need it.")


def test_comfy_backend_is_real():
    assert COMFY_BACKEND == "real", "at least one test must exercise real comfy.utils pack/unpack"


def test_case_a_av_nested_anchor_and_audio_preserved():
    video_shape = (1, 24, 7, 8, 8)
    audio_shape = (1, 32, 2, 20)
    source_v = torch.randn(*video_shape)
    source_a = torch.randn(*audio_shape)
    source = _FakeNested((source_v, source_a))

    den_v = _video_drift(source_v)
    den_a = source_a.clone()
    denoised, _ = _pack([den_v, den_a])

    denoise_v = _center_regenerate_mask(video_shape)
    denoise_a = torch.ones(audio_shape)
    noise_mask = _FakeNested((denoise_v, denoise_a))

    shapes = [video_shape, audio_shape]
    out, inp = _run_hook(source, denoised, noise_mask, shapes, sigma=0.5, strength=1.0)

    preserved = 1.0 - denoise_v
    pres_mask = preserved > 0.5
    out_v = _unpack(out, shapes)[0]
    proc_v = source_v  # video identity through process_latent_in

    residual = (out_v - proc_v).abs()
    assert residual[pres_mask].max().item() < 1e-4

    # Regenerated centre must shift by the same per-channel offset measured on preserved cells.
    for ch in range(video_shape[1]):
        den_vals = den_v[:, ch][pres_mask[:, ch]]
        src_vals = proc_v[:, ch][pres_mask[:, ch]]
        d = (src_vals.mean() - den_vals.mean()).item()
        reg = denoise_v[:, ch] > 0.5
        shift = (out_v[:, ch][reg] - den_v[:, ch][reg]).mean().item()
        assert abs(shift - d) < 1e-4

    assert torch.equal(_unpack(out, shapes)[1], _unpack(inp, shapes)[1])


def test_case_b_video_only_single_stream_unpack_reshape():
    shape = (1, 24, 5, 4, 4)
    source_v = torch.randn(*shape)
    denoise_v = _center_regenerate_mask(shape, inner=1)
    denoised = _single(_video_drift(source_v))
    out, _ = _run_hook(
        source_v,
        denoised,
        denoise_v,
        [shape],
        sigma=0.5,
        strength=1.0,
    )
    out_v = out
    pres = (1.0 - denoise_v) > 0.5
    assert (out_v[pres] - source_v[pres]).abs().max().item() < 1e-4


def test_case_c_sigma_gating_inactive():
    shape = (1, 8, 3, 4, 4)
    source_v = torch.randn(*shape)
    denoise_v = _center_regenerate_mask(shape, inner=1)
    denoised = _single(_video_drift(source_v))
    inp = denoised.clone()
    out, _ = _run_hook(source_v, denoised, denoise_v, [shape], sigma=0.05, end_percent=0.85)
    assert torch.equal(out, inp)


def test_case_d_strength_zero_is_noop():
    shape = (1, 8, 3, 4, 4)
    source_v = torch.randn(*shape)
    denoise_v = _center_regenerate_mask(shape, inner=1)
    denoised = _single(_video_drift(source_v))
    inp = denoised.clone()
    out, _ = _run_hook(source_v, denoised, denoise_v, [shape], strength=0.0)
    assert torch.equal(out, inp)


def test_case_e_min_preserved_skips_empty_frame():
    shape = (1, 4, 3, 4, 4)
    source_v = torch.randn(*shape)
    denoise_v = torch.zeros(shape)
    denoise_v[:, :, 1, :, :] = 1.0  # frame 1 fully regenerate — no preserved cells
    den_v = _video_drift(source_v)
    denoised = _single(den_v)
    out, _ = _run_hook(
        source_v,
        denoised,
        denoise_v,
        [shape],
        min_preserved=0.05,
        strength=1.0,
    )
    out_v = out
    # Frame 1 has zero preserved fraction — left unchanged.
    assert torch.allclose(out_v[:, :, 1], den_v[:, :, 1], atol=1e-6)
    # Frame 0 is fully preserved — should be corrected.
    pres0 = (1.0 - denoise_v[:, :, 0]) > 0.5
    assert (out_v[:, :, 0][pres0] - source_v[:, :, 0][pres0]).abs().max().item() < 1e-4


def test_case_f_exception_safety_no_model_sampling():
    shape = (1, 8, 3, 4, 4)
    source_v = torch.randn(*shape)
    denoise_v = _center_regenerate_mask(shape)
    denoised = _single(_video_drift(source_v))
    hook = make_hook(
        source_v,
        denoise_v,
        1.0,
        "mean",
        True,
        0.05,
        0.0,
        0.85,
        _prepare_mask,
        _pack,
        _unpack,
        shapes_hint=[shape],
    )
    model = _FakeModel([shape], with_sampling=False)
    inp = denoised.clone()
    out = hook({"denoised": inp, "sigma": torch.tensor([0.5]), "model": model})
    assert torch.equal(out, inp)


def test_anchor_video_mean_std_mode():
    shape = (1, 2, 2, 3, 3)
    src = torch.randn(*shape)
    den = src + 0.5
    pres = torch.zeros(shape)
    pres[:, :, :, :2, :] = 1.0
    out, stats = anchor_video(den, src, pres, 1.0, "mean_std", True, 0.01)
    assert stats
    assert (out[:, :, :, :2, :] - src[:, :, :, :2, :]).abs().max().item() < 1e-4


def test_build_preserved_packed_missing_stream_defaults_to_not_preserved():
    v_shape = (1, 4, 2, 2, 2)
    a_shape = (1, 8, 2, 3)
    denoise_v = torch.zeros(v_shape)
    shapes = [v_shape, a_shape]
    preserved = build_preserved_packed(
        _FakeNested((denoise_v,)),
        [v_shape, a_shape],
        _prepare_mask,
        _pack,
        torch.device("cpu"),
    )
    pres_v, pres_a = _unpack(preserved, shapes)
    assert torch.all(pres_v == 1.0)
    assert torch.all(pres_a == 0.0)


def test_split_video_single_stream_is_the_raw_tensor():
    """What the sampler really passes for a non-nested latent."""
    shape = (1, 6, 4, 3, 3)
    video = torch.randn(*shape)
    got, write = split_video(_single(video), [shape], _unpack, _pack)
    assert tuple(got.shape) == shape
    delta = torch.randn_like(got)
    back = write(got + delta)
    assert tuple(back.shape) == shape
    assert torch.allclose(back, video + delta)


def test_split_video_tolerates_a_flat_single_stream():
    """unpack_latents returns a lone stream unreshaped. Not a layout the sampler
    produces today, but the reshape guard costs nothing and survives a core
    change - and write_back must return the SAME flat layout it was given."""
    shape = (1, 6, 4, 3, 3)
    video = torch.randn(*shape)
    flat = video.reshape(1, 1, -1)
    got, write = split_video(flat, [shape], _unpack, _pack)
    assert tuple(got.shape) == shape
    back = write(got + 1.0)
    assert tuple(back.shape) == tuple(flat.shape)
    assert torch.allclose(back.reshape(shape), video + 1.0)


def test_case_g_node_clone_and_no_mask_passthrough():
    from mmx_nodes.latent_color_anchor import MiniMaxH3_LatentColorAnchor

    class _FakePatcher:
        def __init__(self):
            self.model_options: dict = {}
            self._id = object()

        def clone(self):
            cloned = _FakePatcher()
            cloned.model_options = copy.deepcopy(self.model_options)
            return cloned

        def set_model_sampler_post_cfg_function(self, fn):
            import comfy.model_patcher

            self.model_options = comfy.model_patcher.set_model_options_post_cfg_function(
                self.model_options, fn,
            )

    model = _FakePatcher()
    latent = {"samples": torch.randn(1, 8, 3, 4, 4)}
    out = MiniMaxH3_LatentColorAnchor.execute(model, latent)
    assert out[0] is model

    latent_masked = {
        "samples": torch.randn(1, 8, 3, 4, 4),
        "noise_mask": torch.ones(1, 8, 3, 4, 4),
    }
    patched = MiniMaxH3_LatentColorAnchor.execute(model, latent_masked)[0]
    assert patched is not model
    assert "sampler_post_cfg_function" in patched.model_options
    assert len(patched.model_options["sampler_post_cfg_function"]) == 1
