import torch

from mmx_nodes.stitch_back import MiniMaxH3_StitchBack
from mmx_utils import device as device_util
from mmx_utils.transform_types import H3Transform


def test_stitch_back_schema():
    assert MiniMaxH3_StitchBack.define_schema().node_id == "MiniMaxH3_StitchBack"


def _transform():
    return H3Transform(
        boxes=tuple((20.0, 20.0, 60.0, 60.0) for _ in range(2)),
        canvas=(64, 64),
        src_size=(128, 128),
        frames=2,
        weights=(1.0, 1.0),
        detected=(True, True),
        subject_rect=((16.0, 16.0, 32.0, 32.0), (16.0, 16.0, 32.0, 32.0)),
        crop_factor=2.5,
        planner_mode="tv_lp",
    )


def test_stitch_back_execute():
    plate = torch.rand(2, 128, 128, 3)
    crops = torch.rand(2, 64, 64, 3)
    out = MiniMaxH3_StitchBack.execute(
        plate, crops, _transform(), feather_px=32
    )
    images, comp_mask, report = out[0], out[1], out[2]
    assert images.shape == plate.shape
    assert comp_mask.shape == (2, 128, 128)


def test_stitch_back_cpu_fallback_preserves_invariant_5(monkeypatch):
    """An OOM fallback must not weaken the zero-offset contract.

    NOT asserting bit-identity between the baseline and the fallback. On a box
    with a GPU the baseline runs on cuda and the fallback on cpu, and the two
    backends do not produce bit-identical floats for grid_sample and the blend
    reductions. Demanding torch.equal across devices tests the hardware, not us.

    What MUST hold on the fallback path, and is device-independent, is
    invariant 5: outside the mask the output is the LITERAL plate tensor,
    because the composite ends in torch.where(m > 0, blended, plate) — a select,
    not arithmetic. That is the property the whole pack exists to guarantee, so
    that is what this test pins.
    """
    torch.manual_seed(0)
    plate = torch.rand(2, 128, 128, 3)
    crops = torch.rand(2, 64, 64, 3)
    transform = _transform()

    ref_images, _ref_mask = MiniMaxH3_StitchBack.execute(
        plate, crops, transform, feather_px=32
    )[:2]

    real_run = device_util.run_with_cpu_fallback
    attempts: list[torch.device] = []

    def forcing_fallback(work, *, device, label, logger=None):
        def tracked(dev: torch.device):
            attempts.append(dev)
            if len(attempts) == 1 and dev.type != "cpu":
                raise RuntimeError("CUDA out of memory")
            return work(dev)

        accel = torch.device("cuda") if device.type == "cpu" else device
        return real_run(tracked, device=accel, label=label, logger=logger)

    monkeypatch.setattr("mmx_nodes.stitch_back.run_with_cpu_fallback", forcing_fallback)
    out, cmask = MiniMaxH3_StitchBack.execute(
        plate, crops, transform, feather_px=32
    )[:2]

    # the fallback actually fired, and the retry was on CPU
    assert len(attempts) == 2, attempts
    assert attempts[0].type != "cpu" and attempts[1].type == "cpu", attempts

    out = out.detach().cpu()
    cmask = cmask.detach().cpu()
    exterior = cmask <= 0
    assert exterior.any(), "test setup produced no exterior to check"

    # INVARIANT 5 on the fallback path — exact, not approximate
    assert torch.equal(out[exterior], plate[exterior])
    assert torch.isfinite(out).all()

    # and the fallback still computed the same picture, to float tolerance
    assert torch.allclose(out, ref_images.detach().cpu(), atol=1e-4), (
        f"CPU fallback diverged from the accelerator path by "
        f"{(out - ref_images.detach().cpu()).abs().max().item():.2e}"
    )
