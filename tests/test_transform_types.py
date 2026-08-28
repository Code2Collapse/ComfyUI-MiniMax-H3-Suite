from mmx_utils.transform_types import H3Transform


def test_transform_fingerprint_stable():
    t = H3Transform(
        boxes=((1.0, 2.0, 3.0, 4.0),),
        canvas=(512, 512),
        src_size=(1920, 1080),
        frames=1,
        weights=(1.0,),
        detected=(True,),
        subject_rect=((10.0, 10.0, 50.0, 50.0),),
        crop_factor=2.5,
        planner_mode="tv_lp",
    )
    a = t.fingerprint()
    assert a == t.fingerprint()
    assert "nan" not in a.lower()
