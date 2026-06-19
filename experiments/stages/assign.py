"""DVC stage: assign agent (final stage of the chain).

Reads the verify output, runs the assign agent (``assign_agent.pipe``),
serializes the final output dataset, and dumps per-stage metrics JSON. Mirrors
the assign block of ``experiments/benchmark.py``.

NOTE: ``pass_seed=False`` -- in ``benchmark.py`` the assign task-maker is built
without an explicit ``seed`` (it uses the factory default), unlike the other
three agents. We reproduce that exactly so the paper numbers match.
"""

from __future__ import annotations

import assign_agent as step4
from agents.assign_agent import create_assign_agent
from stages._llm_stage import run_chain_stage


def main() -> None:
    run_chain_stage(
        step_module=step4,
        create_agent=create_assign_agent,
        stage_key="assign",
        file_prefix="assign",
        pass_seed=False,
    )


if __name__ == "__main__":
    main()
