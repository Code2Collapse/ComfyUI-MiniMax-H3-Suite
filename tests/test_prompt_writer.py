"""mmx_utils/prompt_writer.py + the MiniMax H3 Prompt Writer v2 node (C2C L8.24, owner A12 s1.5). CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from mmx_utils import llm_http as L  # noqa: E402
from mmx_utils.family_presets import FAMILY_IDS, REPLACEMENT_FAMILIES, build_family_package, parse_six_sections  # noqa: E402
from mmx_utils.prompt_writer import (  # noqa: E402
    PLAN_FIELDS,
    allowed_tags,
    merge_llm_fields,
    parse_llm_json,
    serialise,
    template_plan,
)


def _node():
    from mmx_nodes.prompt_writer import MiniMaxH3_PromptWriter
    return MiniMaxH3_PromptWriter


def _run(**kw):
    out = _node().execute(**kw)
    return out.args if hasattr(out, "args") else out


def test_template_is_deterministic_for_every_family_and_keeps_the_six_sections():
    # INVARIANT: template backend = same inputs, same prompt; all six sections present; the tag lines are the
    # family's own.
    for fam in FAMILY_IDS:
        a, _ = serialise(template_plan(fam, "warm dusk light"))
        b, _ = serialise(template_plan(fam, "warm dusk light"))
        assert a == b
        s = parse_six_sections(a)
        assert all(s[k].strip() for k in s), fam
        base = parse_six_sections(build_family_package(fam)[0])
        assert s["subject_definitions"].startswith(base["subject_definitions"])


def test_replacement_template_states_the_source_performance_and_the_rules():
    # INVARIANT: A12 s1.1 - every replacement prompt names the plate's performance as the authority and carries the
    # mask / temporal / negative rules, even with no LLM.
    for fam in REPLACEMENT_FAMILIES:
        s = parse_six_sections(serialise(template_plan(fam))[0])
        assert "<Video" in s["summary"] and ("frame-accurate" in s["summary"] or "come from <Video 2>" in s["summary"])
        assert "Only the masked region changes" in s["retention_analysis"]
        assert "Must not happen: No identity drift" in s["retention_analysis"]


def test_llm_fields_land_in_their_sections_and_tags_stay_bound():
    # INVARIANT: valid LLM fields are used; the subject/tag bindings are still the template's.
    plan = template_plan("replace_keep_composition")
    tags = allowed_tags(build_family_package("replace_keep_composition")[0])
    reply = {k: "" for k in PLAN_FIELDS}
    reply.update(scene_style="neon night street, anamorphic, teal grade",
                 face_motion="she smiles at the end, lips follow <Video 1>",
                 identity_to_transfer="red curly hair, freckles, green coat")
    merge_llm_fields(plan, reply, tags)
    s = parse_six_sections(serialise(plan)[0])
    assert s["detailed_description"].startswith("The target video: neon night street")
    assert "she smiles at the end" in s["detailed_description"]
    assert "Appearance of <Subject 1>: red curly hair" in s["subject_definitions"]
    assert plan.source["scene_style"] == "llm"
    # an empty LLM field must not erase a template rule
    assert plan.source["negative_constraints"] == "template"


@pytest.mark.parametrize("reply,needle", [
    ("I cannot help with that.", "no JSON object"),
    ('{"scene_style": "a <Picture 3> cameo"}', "<Picture 3>"),
    ('{"scene_style": 42}', "not text"),
])
def test_bad_replies_are_repaired_field_by_field(reply, needle):
    # INVARIANT: invalid JSON, invented references, wrong types -> the template value stays and the repair is listed.
    plan = template_plan("replace_keep_composition", "brief text")
    tags = allowed_tags(build_family_package("replace_keep_composition")[0])
    merge_llm_fields(plan, parse_llm_json(reply), tags)
    assert any(needle in r for r in plan.repairs)
    assert plan.fields["scene_style"] == "brief text"


def test_json_is_found_after_reasoning_and_inside_fences():
    # INVARIANT: thinking models and fenced replies still parse.
    assert parse_llm_json('<think>{"x": 1} hmm</think>```json\n{"scene_style": "a"}\n```') == {"scene_style": "a"}
    assert parse_llm_json('Sure! {"a": "b}"} trailing') == {"a": "b}"}


def test_overlong_description_is_trimmed_at_a_sentence():
    # INVARIANT: detailed_description stays within H3's guide length (500 words) and the trim is reported.
    plan = template_plan("replace_keep_composition", " ".join(["Long sentence about the light."] * 200))
    plan.fields["scene_style"] = " ".join(["Long sentence about the light."] * 200)
    s = parse_six_sections(serialise(plan)[0])
    assert len(s["detailed_description"].split()) <= 500
    assert any("trimmed" in r for r in plan.repairs)


def test_node_template_backend_needs_no_network(monkeypatch):
    # INVARIANT: the default backend never opens a socket.
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    prompt, plan, report = _run(family="replace_keep_composition", brief="rainy window light")
    assert "rainy window light" in prompt
    assert json.loads(plan)["fields"]["scene_style"] == "rainy window light"
    assert report.startswith("backend used: template (no LLM)")


def test_node_falls_back_to_the_template_when_the_backend_is_down():
    # INVARIANT: a dead backend never fails the run by default; the report says what happened.
    prompt, _plan, report = _run(family="replace_keep_composition", brief="b", backend=L.OLLAMA,
                                 base_url="http://127.0.0.1:9", retries=0)
    assert "cannot reach" in report and "deterministic template was used" in report
    assert parse_six_sections(prompt)["summary"]


def test_node_stop_on_failure_raises_a_plain_sentence():
    # INVARIANT: on_failure = stop -> one readable error naming the backend and the reason.
    with pytest.raises(RuntimeError, match="ollama failed - cannot reach"):
        _run(family="replace_keep_composition", backend=L.OLLAMA, base_url="http://127.0.0.1:9", retries=0,
             on_failure="stop with an error")


def test_offline_switch_forces_the_template(monkeypatch):
    # INVARIANT: MMX_PROMPT_WRITER_OFFLINE=1 is the kill switch: no network even with an LLM backend chosen.
    monkeypatch.setenv(L.OFFLINE_ENV, "1")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
    _prompt, _plan, report = _run(family="lipsync", backend=L.OPENAI_COMPAT)
    assert L.OFFLINE_ENV in report


def test_key_value_never_reaches_outputs(monkeypatch):
    # INVARIANT: with a key set and the request failing, neither prompt, plan nor report contains the key.
    secret = "fake-live-key-NEVER-SHOW-0987654321"
    monkeypatch.setenv("C2C_TEST_KEY", secret)
    outs = _run(family="replace_keep_composition", backend=L.OPENAI, base_url="http://127.0.0.1:9/v1", model="m",
                api_key_env="C2C_TEST_KEY", retries=0)
    assert all(secret not in o for o in outs)
    assert "key: from $C2C_TEST_KEY (set)" in outs[2]


def test_fingerprint_holds_no_secret_and_follows_inputs(monkeypatch):
    # INVARIANT: the cache key is built from inputs only (the key's variable NAME, never its value).
    monkeypatch.setenv("C2C_TEST_KEY", "fake-key-SECRET-abcdef")
    N = _node()
    a = N.fingerprint_inputs(family="lipsync", brief="x", api_key_env="C2C_TEST_KEY")
    assert a == N.fingerprint_inputs(family="lipsync", brief="x", api_key_env="C2C_TEST_KEY")
    assert a != N.fingerprint_inputs(family="lipsync", brief="y", api_key_env="C2C_TEST_KEY")
