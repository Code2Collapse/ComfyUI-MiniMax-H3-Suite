import pytest

from mmx_nodes.masked_replace import MiniMaxH3_MaskedReplace


def test_masked_replace_schema():
    schema = MiniMaxH3_MaskedReplace.define_schema()
    assert schema.node_id == "MiniMaxH3_MaskedReplace"
    assert "BasicScheduler" in schema.description


def test_caching_is_left_to_comfyui():
    # A16: the old fingerprint hashed `crops` / `video_latent_mask`, but ComfyUI calls fingerprints with widget values
    # only (linked inputs arrive as None), so it raised, ComfyUI stored NaN and the node re-ran on every queue
    # (measured live, docs/evidence/A16/fingerprints/). Linked inputs are already part of ComfyUI's own cache key.
    assert "fingerprint_inputs" not in MiniMaxH3_MaskedReplace.__dict__
