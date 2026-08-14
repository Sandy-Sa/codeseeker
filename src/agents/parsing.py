"""Shared, model-agnostic response-parsing helpers for the coding agents.

Two robustness problems are handled here, both observed with reasoning models
(DeepSeek-R1-Distill, and expected with Qwen3 / Gemma) served by the project's
vLLM build:

1. **Un-detokenized output.** vLLM (this SIF build) returns the completion as
   GPT-2 byte-level BPE token strings: printable ASCII is intact but whitespace
   bytes are encoded -- space -> 'Ġ' (U+0120), newline -> 'Ċ' (U+010A),
   tab -> 'ĉ', CR -> 'č'. Left undecoded, `\\s`/`splitlines()` find no
   whitespace and the `\\b` word boundary fails next to a digit (e.g. 'Ċ1' has
   no boundary), so every parser collapses or rejects the response. These
   codepoints never occur in real clinical English, so decoding is a safe no-op
   on already-detokenized output.

2. **Tag-less / decorated answers.** The model often omits the requested
   `<answer>` tags and instead writes a markdown list, a `**Answer:** 6, 8`
   line, or codes/IDs after `</think>`. Recovering these avoids a 10x
   temperature-escalating retry storm that silently drops candidates.

Guided decoding (vLLM `guided_regex`) is deliberately NOT used: it constrains the
*entire* generation, which would suppress the `<think>` reasoning these agents
rely on.
"""

from __future__ import annotations

import re

# space, newline, tab, CR as emitted by GPT-2 byte-level BPE (see module docstring).
_BYTE_BPE = str.maketrans({"Ġ": " ", "Ċ": "\n", "ĉ": "\t", "č": "\r"})

THINKING_PATTERN = r"<think>(.*?)</think>"
ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# An explicit answer LABEL in tag-less output, e.g. "**Answer:** 6, 8". A colon
# is required so prose ("...no answer...") does not trigger a false recovery.
_ANSWER_MARKER = re.compile(r"answer\s*\**\s*:", re.IGNORECASE)
# A standalone integer that is NOT part of an alphanumeric ICD code or a decimal,
# so "E11.9" / "J45.909" are skipped while bare candidate IDs ("1", "6, 8",
# "2, 1", and a sentence-final "... ID 2.") are captured. The (?!\.\d) guard
# rejects only true decimals (dot followed by a digit), not a trailing period.
_STANDALONE_INT = re.compile(r"(?<![\w.])\d{1,4}(?![\w])(?!\.\d)")
# An "ID" label glued to its number ("ID0", "ID1, ID7", "IDs: 3", "ID: 5").
# _STANDALONE_INT's (?<![\w.]) lookbehind rejects the digit in "ID0" because
# "D" is a word char, so the label must be stripped first or the model's most
# common answer format parses as *no* IDs at all. The \b keeps real words that
# merely contain "id" + digits intact ("midazolam", "COVID19").
_ID_LABEL = re.compile(r"\bIDs?\s*[:#]?\s*(?=\d)", re.IGNORECASE)
# Every agent template numbers its candidates with jinja's 1-based `loop.index`
# and instructs "If no <terms|codes> are relevant, output ID 0", so 0 is a
# sentinel meaning "nothing applies" and never a selectable candidate.
_NONE_SENTINEL = 0


def decode_byte_bpe(content: str) -> str:
    """Decode GPT-2 byte-level BPE whitespace. Safe/idempotent on clean text."""
    return content.translate(_BYTE_BPE)


def strip_thinking(content: str) -> str:
    """Return the post-reasoning body (everything after the last </think>)."""
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1]
    return re.sub(THINKING_PATTERN, "", content, flags=re.DOTALL)


def extract_int_ids(text: str) -> list[int]:
    """Sorted, de-duplicated standalone integers (candidate IDs) in ``text``.

    The ``ID 0`` sentinel is dropped: candidates are 1-indexed, so 0 carries no
    selection. "ID0" alone therefore yields ``[]`` -- a valid "nothing applies"
    prediction, not a parse failure.
    """
    ids = {int(m) for m in _STANDALONE_INT.findall(_ID_LABEL.sub("", text))}
    ids.discard(_NONE_SENTINEL)
    return sorted(ids)


def parse_id_answer(content: str) -> list[int] | None:
    """Parse a candidate-ID selection from a (reasoning) response.

    Returns the sorted unique IDs. An ``<answer>`` block present but containing
    no IDs (placeholder / "None") yields ``[]`` -- a valid "nothing applies"
    prediction. Returns ``None`` only when no answer could be located at all
    (truncated/malformed), so the caller can raise and let throughster retry.
    """
    content = decode_byte_bpe(content)

    # Reasoning models often draft an <answer> block mid-<think> and then emit
    # the real one at the end, so scan blocks last-first and take the first that
    # actually carries IDs. An answer block that exists but yields none is a
    # deliberate "nothing applies" ([]), not a miss.
    blocks = ANSWER_BLOCK.findall(content)
    if blocks:
        for block in reversed(blocks):
            ids = extract_int_ids(block)
            if ids:
                return ids
        return []

    # No tags: recover an explicit "Answer: ..." line after the reasoning.
    body = strip_thinking(content)
    marker = _ANSWER_MARKER.search(body)
    if marker:
        return extract_int_ids(body[marker.end():])

    return None
