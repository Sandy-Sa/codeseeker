"""Offline unit tests for the analyse-agent parser fallback (no GPU / no vLLM).

These exercise the tag-less recovery path added to ``BaseAnalyseAgent.parser``.
Fixtures are REAL DeepSeek-R1-Distill-Llama-70B responses copied from the failing
run ``.pbs/2026-06-28-15-04-16ef5b07/run.e`` and decoded from the logger's
byte-level BPE artifacts (``Ġ`` -> space, ``Ċ`` -> newline).

Run:
    PYTHONPATH=src:experiments python -m pytest tests/agents/test_analyse_parser.py -q
"""

import pytest

from agents.analyse_agent import BaseAnalyseAgent


def _parser():
    # ``parser`` is a pure method; bypass __init__ (which needs a live client).
    return BaseAnalyseAgent.__new__(BaseAnalyseAgent)


# (content, expected terms that MUST be recovered) -- real captured outputs.
CASES = [
    # 1. dash bullets, "...are:" lead-in, "These conditions ..." trailer
    (
        "from the clinical note are:\n\n- NSTEMI\n- SAH\n- CAD\n- HTN\n"
        "- Hyperlipidemia\n- Obesity\n- Hyperglycemia\n- Hypoalbuminemia\n"
        "- Hypophosphatemia\n- Liver Enzyme Elevation\n\n"
        "These conditions are explicitly documented and relevant to the patient's "
        "current status.",
        {"NSTEMI", "SAH", "CAD", "HTN", "Hyperlipidemia", "Obesity",
         "Hyperglycemia", "Hypoalbuminemia", "Hypophosphatemia",
         "Liver Enzyme Elevation"},
    ),
    # 2. plain newline list (trailing double-spaces), full </think> present
    (
        "...obesity is also a current condition contributing to her health "
        "status. \n</think>\n\nThe current conditions extracted from the clinical "
        "note are:\n\nabdominal pain  \ntachycardia  \npancreatic pseudocyst  \n"
        "SIRS  \npancreatitis  \nobesity",
        {"abdominal pain", "tachycardia", "pancreatic pseudocyst", "SIRS",
         "pancreatitis", "obesity"},
    ),
    # 3. numbered + bold
    (
        "...rm.\n</think>\n\nThe current conditions extracted from the clinical "
        "note are:\n\n1. **Respiratory failure, acute**\n2. **Sepsis**\n"
        "3. **Anastomotic leak**\n\nThese terms are explicitly documented and "
        "relate to the patient's current state during the encounter.",
        {"Respiratory failure, acute", "Sepsis", "Anastomotic leak"},
    ),
    # 4. plain list, no bullets, no lead-in sentence after </think>
    (
        "...and anastomotic leak.\n</think>\n\nGastrointestinal Bleed  \n"
        "Blood Loss Anemia",
        {"Gastrointestinal Bleed", "Blood Loss Anemia"},
    ),
    # 5. abbreviations / dotted abbrevs / hyphen-number terms preserved
    (
        "...meets the criteria.\n</think>\n\nPMR  \nCAD  \nCHF  \nNSTEMI  \n"
        "3-vessel CAD  \nsevere MR  \nleukocytosis  \nanemia  \nRBBB  \n"
        "T wave inversion  \ndiarrhea",
        {"PMR", "CAD", "CHF", "NSTEMI", "3-vessel CAD", "severe MR",
         "leukocytosis", "anemia", "RBBB", "T wave inversion", "diarrhea"},
    ),
]


@pytest.mark.parametrize("content,expected", CASES)
def test_recovers_terms_without_answer_tags(content, expected):
    out = set(_parser().parser(content)["output"])
    assert expected <= out, f"missing: {expected - out}"
    # no explanatory prose should leak through
    assert not any(
        w in t.lower() for t in out for w in ("documented", "criteria", "patient's")
    ), f"prose leaked: {out}"


def _encode_byte_bpe(s: str) -> str:
    """Re-encode whitespace as GPT-2 byte-level BPE, as vLLM actually returns it."""
    return s.replace("\n", "Ċ").replace("\t", "ĉ").replace(" ", "Ġ")


@pytest.mark.parametrize("content,expected", CASES)
def test_recovers_terms_from_undetokenized_byte_bpe(content, expected):
    """Reproduces the real production failure: content arrives un-detokenized
    (spaces='Ġ', newlines='Ċ'). The parser must decode it before recovering."""
    raw = _encode_byte_bpe(content)
    assert "Ġ" in raw and "Ċ" in raw  # sanity: this is the broken form
    out = set(_parser().parser(raw)["output"])
    assert expected <= out, f"missing: {expected - out}"


def test_byte_bpe_answer_tags_parse():
    """A byte-BPE-encoded <answer> block (the path that gave garbage blobs)."""
    content = _encode_byte_bpe(
        "<think>reasoning</think>\n<answer>\nNSTEMI\nsevere MR\n</answer>"
    )
    out = set(_parser().parser(content)["output"])
    assert out == {"NSTEMI", "severe MR"}


def test_well_formed_answer_tags_still_parse():
    """The original <answer> path must be unchanged (truthful to original)."""
    content = (
        "<think>reasoning here</think>\n<answer>\nNSTEMI\nCAD\nHTN\n</answer>"
    )
    out = set(_parser().parser(content)["output"])
    assert out == {"NSTEMI", "CAD", "HTN"}


def test_comma_separated_answer_still_parses():
    content = '<answer>"NSTEMI", "CAD", "HTN"</answer>'
    out = set(_parser().parser(content)["output"])
    assert out == {"NSTEMI", "CAD", "HTN"}


def test_empty_recovery_raises():
    from agents.errors import StructuredError

    with pytest.raises(StructuredError):
        _parser().parser("<think>just reasoning, no list</think>\n\nThe note is unclear.")
