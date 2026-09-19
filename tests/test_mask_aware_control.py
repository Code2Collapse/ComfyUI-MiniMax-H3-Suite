"""The ControlNet residual must stop where the latent mask says stop.

The defect this fixes is in ComfyUI, not here, and it is worth restating
because every assertion below is aimed at it: the core H3 Fun ControlNet adds
its residual to EVERY video row, while latent masking converts the denoise mask
into a per-row timestep telling some of those rows they are already finished.
Two forces on the same rows, pulling opposite ways.

CPU-only and weight-free. The gate arithmetic is pure torch by design, and the
patch is exercised against a fake layout rather than a loaded 5B model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils.mask_aware_control import (  # noqa: E402
    control_gate,
    describe_gate,
    row_values_from_mask,
    soften_rows,
)


def _layout_update(cond_rows: int, video_rows: int) -> torch.Tensor:
    """`layout.img_update`: False for conditioning rows, True for generated."""
    return torch.cat([torch.zeros(cond_rows, dtype=torch.bool),
                      torch.ones(video_rows, dtype=torch.bool)])


def _half_mask(t: int, h: int, w: int) -> torch.Tensor:
    """Top half generates, bottom half is preserved."""
    m = torch.zeros(t, h, w)
    m[:, : h // 2, :] = 1.0
    return m


# ── the row mapping must match the sampler's own ────────────────────────────

def test_row_mapping_matches_comfyui_exactly():
    # THE load-bearing invariant. The gate indexes the same rows the sampler
    # built its per-row timesteps from. If the two orders differ, the control is
    # muted in the WRONG places - which is worse than the original bug, because
    # it looks like the ControlNet is merely weak rather than wrong.
    comfy_fn = pytest.importorskip(
        "comfy.ldm.minimax.model",
        reason="ComfyUI's H3 model is not importable here",
    ).mask_row_values

    torch.manual_seed(0)
    for t, h, w in [(2, 8, 8), (3, 16, 24), (1, 6, 10), (5, 32, 32)]:
        mask = (torch.rand(t, h, w) > 0.5).float()
        ours = row_values_from_mask(mask, t, h, w)
        theirs = comfy_fn(mask, t, h, w)
        if theirs is None:
            continue                      # all-generate; comfy returns None
        assert torch.equal(ours, theirs), f"{t}x{h}x{w} row order differs"


def test_a_patch_counts_as_generated_if_any_pixel_is():
    # amax, not min or mean. A 2x2 patch straddling the mask edge must be
    # treated as generated, or the boundary freezes one patch early and leaves a
    # hard seam exactly where the eye looks.
    mask = torch.zeros(1, 4, 4)
    mask[0, 0, 0] = 1.0                  # one pixel of the first patch
    rows = row_values_from_mask(mask, 1, 4, 4)
    assert rows[0] == 1.0
    assert rows[1:].max() == 0.0


def test_a_mask_smaller_than_the_latent_is_padded_not_rejected():
    rows = row_values_from_mask(torch.ones(1, 6, 6), 1, 8, 8)
    assert rows.shape[0] == 16
    assert rows.min() == 1.0


def test_a_mask_larger_than_the_latent_is_named_not_silently_cropped():
    with pytest.raises(ValueError, match="larger than the latent|cannot be placed"):
        row_values_from_mask(torch.ones(1, 16, 16), 1, 8, 8)


def test_a_wrong_rank_mask_is_named():
    with pytest.raises(ValueError, match=r"\[T,H,W\]"):
        row_values_from_mask(torch.ones(4, 4), 1, 4, 4)


# ── the gate ────────────────────────────────────────────────────────────────

def test_preserved_rows_get_no_control_by_default():
    # THE fix, stated directly.
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    rows = row_values_from_mask(mask, t, h, w)
    update = _layout_update(4, rows.shape[0])
    gate = control_gate(mask, update, t, h, w)
    active = gate[update]
    assert active[rows > 0.5].min() == pytest.approx(1.0)
    assert active[rows < 0.5].max() == pytest.approx(0.0)


def test_conditioning_rows_never_get_control():
    # Keyframes and reference images are not being denoised at all, so a
    # residual on them is pure contradiction whatever the mask says.
    t, h, w = 2, 8, 8
    mask = torch.ones(t, h, w)           # everything generates
    rows = row_values_from_mask(mask, t, h, w)
    update = _layout_update(6, rows.shape[0])
    gate = control_gate(mask, update, t, h, w)
    assert gate[:6].abs().max() == 0.0
    assert gate[6:].min() == pytest.approx(1.0)


def test_no_mask_means_full_control_everywhere_generated():
    # With nothing to conflict with, this must behave exactly like core.
    update = _layout_update(2, 32)
    gate = control_gate(None, update, 2, 8, 8)
    assert gate[2:].min() == pytest.approx(1.0)
    assert gate[:2].max() == 0.0


def test_preserved_strength_one_restores_the_core_behaviour():
    # The escape hatch has to be exact, or "set it to 1 to get the old
    # behaviour" is a lie in the tooltip.
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    update = _layout_update(0, 32)
    gate = control_gate(mask, update, t, h, w, preserved_strength=1.0)
    assert gate.min() == pytest.approx(1.0)


def test_preserved_strength_interpolates():
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    rows = row_values_from_mask(mask, t, h, w)
    update = _layout_update(0, rows.shape[0])
    gate = control_gate(mask, update, t, h, w, preserved_strength=0.25)
    assert gate[rows < 0.5].max() == pytest.approx(0.25)
    assert gate[rows > 0.5].min() == pytest.approx(1.0)


def test_softening_creates_a_ramp_instead_of_a_cliff():
    # A hard gate leaves a visible seam where the control stops.
    t, h, w = 1, 16, 16
    mask = _half_mask(t, h, w)
    update = _layout_update(0, 64)
    hard = control_gate(mask, update, t, h, w, boundary_softness=0.0)
    soft = control_gate(mask, update, t, h, w, boundary_softness=2.0)
    mid_hard = ((hard > 0.05) & (hard < 0.95)).sum()
    mid_soft = ((soft > 0.05) & (soft < 0.95)).sum()
    assert mid_hard == 0, "an unsoftened gate should be binary"
    assert mid_soft > 0, "softening produced no intermediate values"


def test_softening_does_not_blur_across_frames():
    # The blur is 2-D per frame. A 3-D blur would let one frame's mask bleed
    # into the next and smear a moving boundary.
    t, h, w = 3, 8, 8
    mask = torch.zeros(t, h, w)
    mask[1] = 1.0                        # only the middle frame generates
    rows = row_values_from_mask(mask, t, h, w)
    soft = soften_rows(rows, t, h, w, 2.0)
    per_frame = soft.reshape(t, -1)
    assert per_frame[0].max() == 0.0, "frame 0 picked up frame 1's mask"
    assert per_frame[2].max() == 0.0, "frame 2 picked up frame 1's mask"
    assert per_frame[1].min() > 0.0


def test_softening_keeps_the_gate_in_range():
    t, h, w = 2, 16, 16
    mask = _half_mask(t, h, w)
    update = _layout_update(0, 128)
    for softness in (0.0, 0.5, 1.0, 4.0, 16.0):
        gate = control_gate(mask, update, t, h, w, boundary_softness=softness)
        assert gate.min() >= 0.0 and gate.max() <= 1.0, softness
        assert torch.isfinite(gate).all(), softness


def test_a_mismatched_mask_is_named_not_broadcast():
    # A mask describing a different video must say so. Silently broadcasting it
    # would gate the wrong rows and look like a weak ControlNet.
    mask = _half_mask(2, 8, 8)
    update = _layout_update(0, 999)
    with pytest.raises(ValueError, match="different videos|latent rows"):
        control_gate(mask, update, 2, 8, 8)


# ── the report ──────────────────────────────────────────────────────────────

def test_report_counts_the_held_rows():
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    update = _layout_update(4, 32)
    gate = control_gate(mask, update, t, h, w)
    text = describe_gate(gate, update)
    assert "16" in text
    assert "held back" in text


def test_report_says_so_when_nothing_is_held():
    update = _layout_update(0, 32)
    gate = control_gate(None, update, 2, 8, 8)
    assert "fully controlled" in describe_gate(gate, update)


# ── the patch, against a fake layout ────────────────────────────────────────

class _FakeLayout:
    def __init__(self, cond_rows, video_rows, audio_rows, text_rows):
        self.text_rows = text_rows
        n_img = cond_rows + video_rows
        self.img_pos = torch.arange(text_rows, text_rows + n_img)
        self.img_update = _layout_update(cond_rows, video_rows)
        self.audio_pos = torch.arange(text_rows + n_img,
                                      text_rows + n_img + audio_rows)
        self.segments = [(0, text_rows, "text"),
                         (text_rows, text_rows + n_img, "video"),
                         (text_rows + n_img, text_rows + n_img + audio_rows, "audio")]
        self.seq_len = text_rows + n_img + audio_rows


def _apply_gate_like_the_patch(skip, layout, gate, affect_text_rows):
    """The exact sequence MaskAwareControlPatch.after_block performs."""
    skip = skip.clone()
    skip[layout.audio_pos] = 0
    img_pos = layout.img_pos
    skip[img_pos] = skip[img_pos] * gate.to(skip.dtype).unsqueeze(-1)
    if not affect_text_rows:
        for a, b, kind in layout.segments:
            if kind == "text":
                skip[a:b] = 0
    return skip


def test_the_residual_is_zero_exactly_where_the_mask_preserves():
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    rows = row_values_from_mask(mask, t, h, w)
    layout = _FakeLayout(cond_rows=4, video_rows=rows.shape[0],
                         audio_rows=6, text_rows=5)
    gate = control_gate(mask, layout.img_update, t, h, w)

    skip = torch.ones(layout.seq_len, 16)
    out = _apply_gate_like_the_patch(skip, layout, gate, affect_text_rows=False)

    video_pos = layout.img_pos[layout.img_update]
    preserved = video_pos[rows < 0.5]
    generated = video_pos[rows > 0.5]
    assert out[preserved].abs().max() == 0.0, (
        "the control still pushes rows the sampler is holding still"
    )
    assert out[generated].min() == pytest.approx(1.0)
    assert out[layout.audio_pos].abs().max() == 0.0
    assert out[:layout.text_rows].abs().max() == 0.0


def test_affect_text_rows_is_what_changes_the_prompt_rows():
    t, h, w = 2, 8, 8
    mask = torch.ones(t, h, w)
    layout = _FakeLayout(4, 32, 6, 5)
    gate = control_gate(mask, layout.img_update, t, h, w)
    skip = torch.ones(layout.seq_len, 16)

    off = _apply_gate_like_the_patch(skip, layout, gate, affect_text_rows=False)
    on = _apply_gate_like_the_patch(skip, layout, gate, affect_text_rows=True)
    assert off[:5].abs().max() == 0.0
    assert on[:5].abs().max() == pytest.approx(1.0)
    # everything else identical
    assert torch.equal(off[5:], on[5:])


def test_the_ungated_core_behaviour_is_what_we_are_fixing():
    # The counter-test: without the gate, the residual DOES land on preserved
    # rows. If this ever stops being true, ComfyUI fixed it upstream and this
    # node can be retired.
    t, h, w = 2, 8, 8
    mask = _half_mask(t, h, w)
    rows = row_values_from_mask(mask, t, h, w)
    layout = _FakeLayout(4, rows.shape[0], 6, 5)
    ungated = torch.ones(layout.img_update.shape[0])

    skip = torch.ones(layout.seq_len, 16)
    out = _apply_gate_like_the_patch(skip, layout, ungated, affect_text_rows=True)
    preserved = layout.img_pos[layout.img_update][rows < 0.5]
    assert out[preserved].abs().max() == pytest.approx(1.0), (
        "core no longer pushes preserved rows; re-check whether this node is "
        "still needed"
    )


# ── the node ────────────────────────────────────────────────────────────────

def test_the_node_schema_builds_without_h3():
    # A node whose schema needs a model build vanishes from /object_info on any
    # machine that lacks one. That has cost this pack six nodes before.
    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    schema = MiniMaxH3_MaskAwareControlNet.define_schema()
    ids = [getattr(i, "id", None) for i in schema.inputs]
    assert "preserved_strength" in ids
    assert "boundary_softness" in ids
    assert [getattr(o, "id", None) for o in schema.outputs] == ["model", "report"]


def test_the_node_refuses_with_no_hint_at_all():
    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    with pytest.raises(ValueError, match="nothing to steer|no hint"):
        MiniMaxH3_MaskAwareControlNet.execute(
            model=None, control_net=None, vae=None, strength=1.0,
            preserved_strength=0.0, boundary_softness=1.0,
        )


def test_the_report_warns_when_the_fix_is_switched_off():
    # preserved_strength 1.0 silently restores the bug. The report has to say so,
    # or a user who set it while experimenting never finds their way back.
    import mmx_nodes.mask_aware_control as mod

    captured = {}

    class _FakePatch:
        def __init__(self, *a, **k):
            captured.update(k)

        def register(self, _model):
            captured["registered"] = True

    class _FakeModel:
        def clone(self):
            return self

    original = mod.MaskAwareControlPatch
    mod.MaskAwareControlPatch = _FakePatch
    try:
        out = mod.MiniMaxH3_MaskAwareControlNet.execute(
            model=_FakeModel(), control_net=object(), vae=object(), strength=1.0,
            preserved_strength=1.0, boundary_softness=0.0,
            control_video=torch.zeros(1, 8, 8, 3),
        )
    finally:
        mod.MaskAwareControlPatch = original

    report = out[1] if isinstance(out, tuple) else out.result[1]
    assert "ignored" in report.lower()
    assert captured["registered"] is True


# ── the front-end ───────────────────────────────────────────────────────────

def test_the_gate_plot_is_present_and_targets_this_node():
    # WEB_DIRECTORY is "web" and ComfyUI loads every file in it, so sitting in
    # that directory IS the wiring. What can still rot is the node id.
    js = (PACK / "web" / "w7_mask_aware_control.js")
    assert js.is_file()
    src = js.read_text(encoding="utf-8")
    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    node_id = MiniMaxH3_MaskAwareControlNet.define_schema().node_id
    assert f'"{node_id}"' in src, (
        f"the gate plot targets a different node id than {node_id}"
    )


def test_the_gate_plot_reads_widgets_that_exist():
    # The curve is drawn from three widget values. A rename leaves the plot
    # reading undefined and drawing a flat line, which looks like the node
    # doing nothing rather than the plot being broken.
    import re

    from mmx_nodes.mask_aware_control import MiniMaxH3_MaskAwareControlNet

    src = (PACK / "web" / "w7_mask_aware_control.js").read_text(encoding="utf-8")
    names = {getattr(i, "id", None)
             for i in MiniMaxH3_MaskAwareControlNet.define_schema().inputs}
    for widget in re.findall(r'read\("([^"]+)"', src):
        assert widget in names, (
            f"the gate plot reads widget {widget!r}, which this node does not have"
        )
    assert {"preserved_strength", "boundary_softness"} <= set(
        re.findall(r'read\("([^"]+)"', src))


def test_the_plot_module_has_no_vue_and_chains_teardown():
    import re

    src = (PACK / "web" / "w7_mask_aware_control.js").read_text(encoding="utf-8")
    # An IMPORT of Vue, not the word. The module's own docstring says "no Vue",
    # so a substring check on that is a test that fails for being right - which
    # is exactly what the first version of this did.
    imports = re.findall(r'(?:from|import)\s*\(?\s*["\']([^"\']+)["\']', src)
    vue = [spec for spec in imports if "vue" in spec.lower()]
    assert not vue, f"Vue is imported here: {vue}"
    assert "chainOnRemoved" in src, "an unchained onRemoved leaks the observer"
    assert "rafThrottle" in src, "an unthrottled repaint runs on every widget pixel"
