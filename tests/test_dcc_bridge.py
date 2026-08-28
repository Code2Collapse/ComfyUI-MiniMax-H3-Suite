import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from mmx_utils.dcc_bridge import (
    build_retention_payload,
    load_exr_sequence,
    read_exr_plane,
    resolve_exr_paths,
    retention_json_text,
)


def test_retention_schema():
    payload = build_retention_payload(sim_type="fire", interaction_mode="attribute_transfer", fps=24.0)
    text = retention_json_text(payload)
    data = json.loads(text)
    assert data["video_label"] == "<Video 1>"
    assert data["sim_type"] == "fire"
    assert data["interaction_mode"] == "attribute_transfer"
    assert any("camera:" in line for line in data["retention_analysis_lines"])


def test_missing_exr_raises_human_sentence():
    with pytest.raises(Exception, match="No EXR frames matched"):
        resolve_exr_paths("/nonexistent/path/seq_####.exr")


@pytest.fixture
def hdr_exr_file(tmp_path):
    oiio = pytest.importorskip("OpenImageIO")
    arr = np.ones((8, 8, 3), dtype=np.float32) * 4.0
    path = tmp_path / "frame_0001.exr"
    out = oiio.ImageOutput.create(str(path))
    assert out
    spec = oiio.ImageSpec(8, 8, 3, oiio.FLOAT)
    out.open(str(path), spec)
    out.write_image(arr)
    out.close()
    return path


def test_exr_values_above_one_survive(hdr_exr_file):
    plane = read_exr_plane(str(hdr_exr_file))
    assert float(plane.max()) > 1.0


def test_exr_more_than_three_channels_raises(hdr_exr_file):
    oiio = pytest.importorskip("OpenImageIO")
    arr = np.ones((8, 8, 4), dtype=np.float32)
    path = hdr_exr_file.parent / "frame_4ch.exr"
    out = oiio.ImageOutput.create(str(path))
    assert out
    spec = oiio.ImageSpec(8, 8, 4, oiio.FLOAT)
    out.open(str(path), spec)
    out.write_image(arr)
    out.close()
    with pytest.raises(Exception, match="4 channels"):
        read_exr_plane(str(path))


def test_load_exr_sequence_tensor(hdr_exr_file):
    video = load_exr_sequence(str(hdr_exr_file.parent))
    assert video.shape[0] == 1
    assert float(video.max()) > 1.0
    assert isinstance(video, torch.Tensor)
