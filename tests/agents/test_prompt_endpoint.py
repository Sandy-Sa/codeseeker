"""Offline tests for endpoint-dependent prompt rendering (no GPU / no vLLM).

Every agent template ends its user turn with a literal `<think>` to prime
DeepSeek-R1's reasoning on the raw `completions` path. On `chat/completions`
vLLM applies the model's own chat template, which already appends
`<|Assistant|><think>`, so ours must be stripped -- otherwise a stray `<think>`
lands *inside* the user turn.

Run:
    PYTHONPATH=src:experiments python -m pytest tests/agents/test_prompt_endpoint.py -q
"""

import types

import pytest

from agents.base import HfBaseAgent


class _FakePrompt:
    """Stands in for prompt_poet.Prompt (only `.messages` / `.string` are used)."""

    def __init__(self, messages: list[dict[str, str]]):
        self.messages = messages

    @property
    def string(self) -> str:
        return "\n".join(m["content"] for m in self.messages)


def _client(endpoint: str):
    return types.SimpleNamespace(endpoint=endpoint)


MESSAGES = [
    {"role": "system", "content": "You are a medical coding assistant."},
    {"role": "user", "content": "Clinical Note: x\n\nReason step by step.\n<think>"},
]


def test_chat_completions_strips_trailing_think():
    out = HfBaseAgent.prompt_messages_or_string(
        _client("chat/completions"), _FakePrompt([dict(m) for m in MESSAGES])
    )
    assert isinstance(out, list)
    assert out[-1]["content"].endswith("Reason step by step.")
    assert "<think>" not in out[-1]["content"]
    # the system turn is untouched
    assert out[0]["content"] == MESSAGES[0]["content"]


def test_completions_keeps_trailing_think():
    out = HfBaseAgent.prompt_messages_or_string(
        _client("completions"), _FakePrompt([dict(m) for m in MESSAGES])
    )
    assert isinstance(out, str)
    assert out.endswith("<think>")


def test_chat_path_does_not_mutate_the_prompt_object():
    prompt = _FakePrompt([dict(m) for m in MESSAGES])
    HfBaseAgent.prompt_messages_or_string(_client("chat/completions"), prompt)
    # rendering must be repeatable: the underlying messages keep their <think>
    assert prompt.messages[-1]["content"].endswith("<think>")


@pytest.mark.parametrize(
    "content",
    [
        "no marker here",
        "<think> in the middle of the turn",
        "trailing think word without tags",
    ],
)
def test_only_a_trailing_tag_is_stripped(content):
    out = HfBaseAgent.prompt_messages_or_string(
        _client("chat/completions"),
        _FakePrompt([{"role": "user", "content": content}]),
    )
    assert out[-1]["content"] == content


def test_real_templates_all_end_in_think():
    """Guards the assumption above against template drift."""
    from jinja2 import Environment, FileSystemLoader

    from agents.base import PATH_TO_TEMPLATES

    env = Environment(loader=FileSystemLoader(PATH_TO_TEMPLATES), autoescape=False)
    in_use = [
        "analyse_agent/strict_v3",
        "locate_agent/locate_few_terms_v1",
        "verify_agent/few_per_term_v2",
        "assign_agent/reasoning_v6",
    ]
    for name in in_use:
        raw, _, _ = env.loader.get_source(env, f"{name}.yml.j2")
        assert raw.rstrip().endswith("<think>"), name
