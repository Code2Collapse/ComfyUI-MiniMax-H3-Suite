from mmx_utils.vitpose_plate import VitPoseUnavailableError, resolve_vitpose_paths


def test_resolve_vitpose_paths_returns_tuple():
    vit, yolo = resolve_vitpose_paths()
    assert vit is None or vit.endswith(".onnx")
    assert yolo is None or yolo.endswith(".onnx")


def test_vitpose_unavailable_error_message():
    err = VitPoseUnavailableError("weights missing")
    assert "weights missing" in str(err)
