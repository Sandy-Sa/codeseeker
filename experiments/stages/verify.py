"""DVC stage: verify agent.

Reads the locate output, runs the verify agent (``verify_agent.pipe``),
serializes the full output dataset for the assign stage, and dumps per-stage
metrics JSON. Mirrors the verify block of ``experiments/benchmark.py``.
"""

from __future__ import annotations

import verify_agent as step3
from agents.verify_agent import create_verify_agent
from stages._llm_stage import run_chain_stage


def main() -> None:
    run_chain_stage(
        step_module=step3,
        create_agent=create_verify_agent,
        stage_key="verify",
        file_prefix="verify",
        pass_seed=True,
    )


if __name__ == "__main__":
    main()
