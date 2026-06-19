"""DVC stage: locate agent.

Reads the analyse output, runs the locate agent (``locate_agent.pipe``),
serializes the full output dataset for the verify stage, and dumps per-stage
metrics JSON. Mirrors the locate block of ``experiments/benchmark.py``.
"""

from __future__ import annotations

import locate_agent as step2
from agents.locate_agent import create_locate_agent
from stages._llm_stage import run_chain_stage


def main() -> None:
    run_chain_stage(
        step_module=step2,
        create_agent=create_locate_agent,
        stage_key="locate",
        file_prefix="locate",
        pass_seed=True,
    )


if __name__ == "__main__":
    main()
