"""Offline tests for transient-failure handling in HfBaseAgent (no GPU / no vLLM).

Job 176217145 died at row 518/550 of `locate`, ~58 minutes in, on a single bare
`TimeoutError`: throughster's _batch_decorator only converts
StructuredResponseError into a per-request result, so anything else escapes the
anyio task group, kills the datasets.map(num_proc>1) worker and discards every
row the stage had already finished. These tests pin the bounded retry that keeps
one transient fault from costing the whole stage.

Run:
    PYTHONPATH=src:experiments python -m pytest tests/agents/test_batch_retry.py -q
"""

import pytest

from agents.base import HfBaseAgent


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Collapse the backoff so the suite stays instant."""
    monkeypatch.setattr("agents.base.time.sleep", lambda _s: None)


def _agent(side_effects):
    """An HfBaseAgent whose batch_call replays ``side_effects`` in order."""
    agent = HfBaseAgent.__new__(HfBaseAgent)
    calls = []

    def batch_call(requests):
        calls.append(requests)
        effect = side_effects[len(calls) - 1]
        if isinstance(effect, Exception):
            raise effect
        return effect

    agent.batch_call = batch_call
    agent.calls = calls
    return agent


def test_transient_failure_is_retried_and_recovers():
    # the exact exception that ended job 176217145
    agent = _agent([TimeoutError(), ["ok"]])
    assert agent._batch_call_with_retries([{"p": 1}]) == ["ok"]
    assert len(agent.calls) == 2


def test_success_on_first_attempt_does_not_retry():
    agent = _agent([["ok"]])
    assert agent._batch_call_with_retries([{"p": 1}]) == ["ok"]
    assert len(agent.calls) == 1


def test_persistent_failure_still_raises_after_bounded_retries():
    # a systematically broken stage must still fail loudly, not spin
    agent = _agent([TimeoutError() for _ in range(HfBaseAgent._BATCH_RETRIES)])
    with pytest.raises(TimeoutError):
        agent._batch_call_with_retries([{"p": 1}])
    assert len(agent.calls) == HfBaseAgent._BATCH_RETRIES


def test_retries_replay_the_same_requests():
    # identical requests => same throughster cache keys => the sub-requests that
    # already succeeded replay from disk and only the failed one hits the GPU
    requests = [{"p": 1}, {"p": 2}]
    agent = _agent([ConnectionError(), requests])
    agent._batch_call_with_retries(requests)
    assert agent.calls == [requests, requests]
