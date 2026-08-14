"""Offline tests for locate/verify/assign ID parsing (no GPU / no vLLM).

Fixtures are REAL responses copied from .pbs/2026-06-29-12-34-c1b65ff4/run.e,
kept in their on-the-wire byte-level BPE form (space='Ġ', newline='Ċ') -- exactly
what crippled locate (252) and verify (238) with retry storms. The parser must
decode and recover IDs from all of these.

Run:
    PYTHONPATH=src:experiments python -m pytest tests/agents/test_id_parsers.py -q
"""

import pytest

from agents.errors import StructuredError
from agents.locate_agent import LocateAgent
from agents.verify_agent import VerifyAgent
from agents.assign_agent import AssignAgent
from agents.parsing import parse_id_answer


def _p(cls):
    return cls.__new__(cls)  # parser is a pure method


# --- shared parse_id_answer behaviour (real byte-BPE fixtures) --------------
# (raw on-the-wire content, expected ids). None expected => unparseable -> retry.
ID_CASES = [
    # <answer> with single id, byte-BPE encoded (the '\b' next to 'Ċ1' bug)
    ("...necrosis.Ċ</think>ĊĊ...uninfectedĠnecrosis.ĊĊ<answer>Ċ1Ċ</answer>", [1]),
    # comma list inside answer
    ("reasoning.Ċ</think>ĊĊ...codesĠare:ĊĊ<answer>Ċ1,2Ċ</answer>", [1, 2]),
    # reversed comma list "2, 1" -> sorted unique
    ("...metastasis.ĊĊ<answer>Ċ2,Ġ1Ċ</answer>", [1, 2]),
    # tag-less "**Answer:** 1" after </think>
    ("...Bell'sĠpalsy.Ċ</think>ĊĊTheĠpatient...palsy.ĊĊ**Answer:**Ġ1", [1]),
    # tag-less "**Answer:** 6, 8"
    ("...findingsĠare:ĊĊ**Answer:**Ġ6,Ġ8", [6, 8]),
    # IDs written in prose inside the answer block: "ID 1 and ID 2"
    ("ok.ĊĊ<answer>ĊTheĠrelevantĠcodesĠareĠIDĠ1ĠandĠIDĠ2.Ċ</answer>", [1, 2]),
    # answer block present but no IDs (placeholder / "None") -> [] (not a failure)
    ("done.Ċ</think>ĊĊ<answer>None</answer>", []),
    # ICD codes must NOT be mistaken for IDs (E11.9, J45.909 -> skipped); only
    # the standalone selection id is taken.
    ("...analysis.Ċ</think>ĊĊTheĠcodeĠisĠE11.9.ĊĊ<answer>Ċ3Ċ</answer>", [3]),
    # truncated / no answer at all -> None -> caller raises
    ("...stillĠreasoningĠwithoutĠanyĠconclusionĠyet", None),
    # --- "ID" glued to its number (.pbs/2026-08-14-10-22-68d95d59/run.e) -----
    # The dominant answer format in job 176217145. _STANDALONE_INT's (?<![\w.])
    # lookbehind rejected the digit after "D", so all of these parsed as no IDs
    # and burned retries.
    ("...noneĠapply.ĊĊ<answer>ĊID0Ċ</answer>", []),
    ("...esophagitis.ĊĊ<answer>ID1,ĠID7</answer>", [1, 7]),
    ("...</think>ĊĊ<answer>ID0</answer>", []),
    ("...diabetes.ĊĊ**Answer:**ĠID1,ĠID5", [1, 5]),
    ("...hematuria.ĊĊ**Answer:**ĠID4", [4]),
    ("...PICCĠinfection.ĊĊ<answer>5,6</answer>", [5, 6]),
    # "ID 0" is the template's "nothing applies" sentinel (candidates are
    # 1-indexed via jinja loop.index), so it must never become candidate 0.
    ("...noĠtermsĠapply.Ċ</think>ĊĊ<answer>ĠID0Ġ</answer>", []),
    ("...mixed.ĊĊ<answer>ID0,ĠID3</answer>", [3]),
    # a draft <answer> inside the reasoning must not shadow the final one
    ("<think>ĊmaybeĠ<answer>9</answer>Ċ</think>ĊĊ<answer>2</answer>", [2]),
    # words containing "id"+digits must survive the label strip
    ("...onĠmidazolam,ĠCOVID19Ġnegative.ĊĊ<answer>2</answer>", [2]),
]


@pytest.mark.parametrize("content,expected", ID_CASES)
def test_parse_id_answer(content, expected):
    assert parse_id_answer(content) == expected


# --- per-agent parser semantics --------------------------------------------
def test_verify_recovers_byte_bpe_single():
    out = _p(VerifyAgent).parser("x.Ċ</think>ĊĊ<answer>Ċ1Ċ</answer>")["output"]
    assert out == [1]


def test_verify_empty_answer_block_is_valid_none():
    # "none of the candidates apply" -> [] kept, NOT retried
    out = _p(VerifyAgent).parser("x.Ċ</think>ĊĊ<answer>None</answer>")["output"]
    assert out == []


def test_verify_truncated_raises():
    with pytest.raises(StructuredError):
        _p(VerifyAgent).parser("stillĠthinking,ĠnoĠanswer")


def test_assign_empty_answer_block_is_empty_prediction():
    out = _p(AssignAgent).parser("x.Ċ</think>ĊĊ<answer> </answer>")["output"]
    assert out == []


def test_assign_recovers_comma_list():
    out = _p(AssignAgent).parser("x.Ċ</think>ĊĊ<answer>Ċ1,Ġ4,Ġ7Ċ</answer>")["output"]
    assert out == [1, 4, 7]


def test_locate_recovers_tagless_answer_line():
    out = _p(LocateAgent).parser("x.Ċ</think>ĊĊ**Answer:**Ġ6,Ġ8")["output"]
    assert out == [6, 8]


def test_locate_empty_answer_block_is_valid_none():
    # "ID 0" / "None" is the sentinel the locate template asks for when no term
    # applies -- a valid negative, NOT a parse miss. Retrying it made the agent
    # unable to express a negative at all (job 176217145).
    assert _p(LocateAgent).parser("x.Ċ</think>ĊĊ<answer>None</answer>")["output"] == []
    assert _p(LocateAgent).parser("x.Ċ</think>ĊĊ<answer>ID0</answer>")["output"] == []


def test_locate_truncated_raises():
    # no answer at all (truncated/malformed) is still a parse miss -> retry
    with pytest.raises(StructuredError):
        _p(LocateAgent).parser("stillĠthinking,ĠnoĠanswer")


def test_locate_recovers_glued_id_labels():
    out = _p(LocateAgent).parser("x.Ċ</think>ĊĊ<answer>ID1,ĠID7</answer>")["output"]
    assert out == [1, 7]
