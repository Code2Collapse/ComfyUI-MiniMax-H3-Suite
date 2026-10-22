"""H3 Tagged references (ported from ethanfel/ComfyUI-MiniMaxH3-Context-Loop, GPL-3.0): the tag registry, prompt
compilation with strict / soft / disabled compliance and H3's 9 / 3 / 3 capacity, and the native labels checked against
core MiniMaxH3ReferenceToVideo's own presentation order."""
from __future__ import annotations

import pytest
import torch

from mmx_utils import tagged_refs as TR
from mmx_nodes import tagged_refs as N


def pic(v=0.5):
    return torch.full((1, 32, 48, 3), v)


def vid(n=8, v=0.3):
    return torch.full((n, 32, 48, 3), v)


def snd(v=0.1, n=8000):
    return {"waveform": torch.full((1, 2, n), v), "sample_rate": 32000}


def out_of(o):
    return o.result if hasattr(o, "result") else o.args


def chain():
    r = out_of(N.MiniMaxH3_TaggedPicture.execute(pic(0.1), "hero_face"))[0]
    r = out_of(N.MiniMaxH3_TaggedPicture.execute(pic(0.2), "@villain", previous=r))[0]
    r = out_of(N.MiniMaxH3_TaggedVideo.execute(vid(), "performance", "", audio=snd(), previous=r))[0]
    r = out_of(N.MiniMaxH3_TaggedVideo.execute(vid(v=0.6), "walk", "", previous=r))[0]
    r = out_of(N.MiniMaxH3_TaggedAudio.execute(snd(0.4), "voice", previous=r))[0]
    return r


def test_tags_are_normalised_and_unique():
    assert TR.normalize_tag("@hero-face") == "hero-face"
    for bad in ("", "1abc", "has space", "a" * 65, "é"):
        with pytest.raises(TR.TaggedRefError):
            TR.normalize_tag(bad)
    r = chain()
    assert [e["tag"] for e in r["entries"]] == ["hero_face", "villain", "performance", "walk", "voice"]
    assert r["entries"][2]["audio_tag"] == "performance_audio" and "audio_tag" not in r["entries"][3]
    with pytest.raises(TR.TaggedRefError, match="already registered"):
        N.MiniMaxH3_TaggedAudio.execute(snd(), "performance_audio", previous=r)
    with pytest.raises(TR.TaggedRefError, match="must differ"):
        N.MiniMaxH3_TaggedVideo.execute(vid(), "x_take", "x_take", audio=snd())


def test_only_mentioned_references_are_sent_with_native_labels():
    r = chain()
    text, b, report = TR.compile_prompt(r, "<Subject 1> has the face of @villain, moves like @performance, "
                                           "and speaks with @voice.")
    assert text == ("<Subject 1> has the face of <Picture 1>, moves like <Video 1>, and speaks with <Audio 2>.")
    assert [e["tag"] for e in b["pictures"]] == ["villain"]      # @hero_face is registered but not used
    assert "@performance_audio -> <Audio 1>" in report and "@voice -> <Audio 2>" in report
    ins = TR.ref2va_inputs(b)
    assert list(ins) == ["ref_images", "ref_videos", "ref_video_audios", "ref_audios"]
    assert torch.equal(ins["ref_images"]["ref_image_1"], r["entries"][1]["value"])
    assert set(ins["ref_video_audios"]) == {"ref_video_audio_1"}


def test_soundtrack_tag_activates_the_pair():
    text, b, _ = TR.compile_prompt(chain(), "Keep the rhythm of @performance_audio.")
    assert text == "Keep the rhythm of <Audio 1>." and [e["tag"] for e in b["videos"]] == ["performance"]


def test_numbering_follows_chain_order():
    text, _b, _ = TR.compile_prompt(chain(), "@villain then @hero_face; @walk and @performance.")
    assert text == "<Picture 2> then <Picture 1>; <Video 2> and <Video 1>."


def test_unknown_tags_and_compliance_modes():
    r = chain()
    with pytest.raises(TR.TaggedRefError, match="not a registered"):
        TR.compile_prompt(r, "@heroface")
    text, _b, report = TR.compile_prompt(r, "@heroface and @villain", "soft")
    assert text == "@heroface and <Picture 1>" and "warning: @heroface is not a registered reference" in report
    text, b, report = TR.compile_prompt(r, "@heroface and @villain", "disabled")
    assert text == "@heroface and @villain" and len(b["pictures"]) == 1 and "left in the prompt" in report
    assert TR.compile_prompt(r, "an email like a@b.c is not a tag")[0] == "an email like a@b.c is not a tag"


def test_capacity():
    r = None
    for i in range(10):
        r = TR.append(r, "picture", f"p{i}", pic(i / 10), TR.tensor_fingerprint(pic(i / 10)))
    prompt = " ".join(f"@p{i}" for i in range(10))
    with pytest.raises(TR.TaggedRefError, match="takes 9"):
        TR.compile_prompt(r, prompt)
    text, b, report = TR.compile_prompt(r, prompt, "soft")
    assert len(b["pictures"]) == 9 and text.endswith("<Picture 9> @p9") and "dropped" in report


def test_registry_is_refused_from_elsewhere():
    with pytest.raises(TR.TaggedRefError, match="H3 Tagged"):
        TR.compile_prompt({"entries": []}, "@x")
    with pytest.raises(TR.TaggedRefError):
        N.MiniMaxH3_TaggedVideo.execute(vid(3), "short")          # H3 needs at least 5 frames
    with pytest.raises(TR.TaggedRefError):
        N.MiniMaxH3_TaggedAudio.execute({"waveform": torch.zeros(2, 10)}, "bad")


def test_fingerprint_follows_content_not_identity():
    a = TR.tensor_fingerprint(pic(0.5))
    assert a == TR.tensor_fingerprint(pic(0.5)) and a != TR.tensor_fingerprint(pic(0.6))
    assert TR.tensor_fingerprint(torch.zeros(2, 3, dtype=torch.bfloat16))


class FakeClip:
    """Records what core's Reference to Video hands the text encoder."""

    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kw):
        self.calls.append((prompt, kw.get("minimax_ref_items")))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


def test_labels_match_core_reference_to_video_order():
    core = pytest.importorskip("comfy_extras.nodes_minimax_h3")
    clip = FakeClip()
    r = chain()
    out = out_of(N.MiniMaxH3_TaggedReferenceToVideo.execute(
        clip, r, "@voice, @walk, @performance, @hero_face, @villain", width=64, height=32, length=22))
    prompt, items = clip.calls[-1]
    # core presents pictures, then each video (its soundtrack's <Audio j> first), then standalone audio
    kinds = [it["type"] for it in items]
    assert kinds == ["image", "image", "audio", "video", "video", "audio"]
    assert prompt == "<Audio 2>, <Video 2>, <Video 1>, <Picture 1>, <Picture 2>"
    labels = []
    counters = {"image": 0, "video": 0, "audio": 0}
    for k in kinds:
        counters[k] += 1
        labels.append({"image": "Picture", "video": "Video", "audio": "Audio"}[k] + f" {counters[k]}")
    assert labels == ["Picture 1", "Picture 2", "Audio 1", "Video 1", "Video 2", "Audio 2"]
    assert out[2] == prompt and "@performance_audio" not in out[2]
    assert core.MiniMaxH3ReferenceToVideo is N._core_ref2va()
