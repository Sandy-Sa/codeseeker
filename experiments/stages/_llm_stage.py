"""Shared driver for the non-retrieval LLM chain stages (locate/verify/assign).

These three stages have an identical shape -- the only differences are the step
module, the agent factory, the params key, the metrics prefix, and whether the
task-maker is built with an explicit ``seed`` (assign is not, mirroring
``experiments/benchmark.py``). Keeping them on one code path guarantees the DVC
stages stay byte-for-byte faithful to the in-memory benchmark used to produce the
paper numbers.

``analyse`` is intentionally NOT routed through here: it additionally builds the
embedded-Qdrant retrieval service and calls ``pipe`` with retrieval kwargs.
"""

from __future__ import annotations

import argparse
import pathlib
import typing as typ

import datasets

import utils as exp_utils
from stages import _config


def run_chain_stage(
    *,
    step_module: typ.Any,
    create_agent: typ.Callable[..., typ.Any],
    stage_key: str,
    file_prefix: str,
    pass_seed: bool,
) -> None:
    """Parse CLI args and run one ``locate``/``verify``/``assign`` stage.

    Mirrors the corresponding block of ``experiments/benchmark.py``:
    ``load_from_disk`` the previous stage -> build agent + trie -> ``step.pipe``
    -> ``save_to_disk`` the full dataset -> ``evaluate_and_dump_metrics``.

    Args:
        step_module: ``experiments/{locate,verify,assign}_agent.py`` (exposes ``pipe``).
        create_agent: matching ``create_*_agent`` factory.
        stage_key: key into ``params.yaml -> agents`` (e.g. ``"locate"``).
        file_prefix: metrics file prefix (``locate``/``verify``/``assign``).
        pass_seed: forward ``seed`` to the task-maker. ``assign`` passes ``False``
            so the agent uses the factory default seed (42), as in ``benchmark.py``.
    """
    parser = argparse.ArgumentParser(description=f"{file_prefix} stage")
    parser.add_argument("--input", required=True, help="prev stage output (load_from_disk)")
    parser.add_argument("--output", required=True, help="save_to_disk target dir")
    parser.add_argument("--metrics-dir", required=True, help="where to dump metrics")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    args = parser.parse_args()

    cfg = _config.load_config(args.params)
    metrics_dir = pathlib.Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(args.input)
    xml_trie = exp_utils.build_icd_trie(year=cfg.icd.year)
    eval_trie = _config.build_eval_trie(xml_trie)

    agent_cfg = cfg.agents[stage_key]
    sampling_params = cfg.sampling_params()
    if agent_cfg.max_tokens is not None:
        # Per-stage output cap so this stage's prompts fit the model context
        # (see AgentConfig.max_tokens). Faithful where the global cap already
        # fits -- only narrows it for the large-prompt stages on a small model.
        sampling_params["max_tokens"] = agent_cfg.max_tokens
    agent = create_agent(
        agent_type=agent_cfg.agent_type,
        prompt_name=agent_cfg.prompt_name,
        sampling_params=sampling_params,
    )

    maker_kwargs: dict[str, typ.Any] = {
        "init_client_fn": exp_utils._init_client_fn(**cfg.base_model),
    }
    if pass_seed:
        maker_kwargs["seed"] = cfg.sampling.seed
    task_maker = agent(**maker_kwargs)

    eval_data = step_module.pipe(
        agent=task_maker,
        dataset=dataset,
        trie=xml_trie,
        num_workers=cfg.runtime.num_workers,
        batch_size=cfg.runtime.batch_size,
        seed=cfg.sampling.seed,
    )

    eval_data.save_to_disk(args.output)

    exp_utils.evaluate_and_dump_metrics(
        eval_data=eval_data,
        trie=eval_trie,
        dump_path=metrics_dir,
        file_prefix=file_prefix,
    )
