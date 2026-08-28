import torch

from mmx_nodes.frame_handles import MiniMaxH3_FrameHandles


def test_frame_handles_h3_pad():
    images = torch.rand(20, 32, 32, 3)
    out = MiniMaxH3_FrameHandles.execute(images, handle_frames=0, padding_mode="H3 (17n+5)")
    imgs, audio, fc, report = out[0], out[1], out[2], out[3]
    assert fc == 22
    assert imgs.shape[0] == 22
    assert "22" in report
