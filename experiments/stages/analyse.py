"""DVC stage: analyse agent.

Reads the prepped dataset, runs the analyse agent + retrieval (`step1.pipe`),
serializes the full output dataset for the next stage, and dumps per-stage
metrics JSON.
"""

from __future__ import annotations

import argparse
import pathlib

import datasets

import analyse_agent as step1
from agents.analyse_agent import create_analyse_agent
import utils as exp_utils
from retrieval.qdrant_search import client as qdrant_client
from stages import _config


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse stage")
    parser.add_argument("--input", required=True, help="prep output (load_from_disk)")
    parser.add_argument("--output", required=True, help="save_to_disk target dir")
    parser.add_argument("--metrics-dir", required=True, help="where to dump metrics")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    args = parser.parse_args()

    cfg = _config.load_config(args.params)
    metrics_dir = pathlib.Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(args.input)
    xml_trie = exp_utils.build_icd_trie(year=cfg.icd.year)
    eval_trie = _config.build_eval_trie(xml_trie, all_codes=cfg.retrieval.all_codes)

    qdrant_service = qdrant_client.QdrantSearchService(
        local_path=cfg.qdrant_local_path()
    )

    agent_cfg = cfg.agents["analyse"]
    analyse_agent = create_analyse_agent(
        agent_type=agent_cfg.agent_type,
        prompt_name=agent_cfg.prompt_name,
        sampling_params=cfg.sampling_params(),
    )
    task_maker = analyse_agent(
        init_client_fn=exp_utils._init_client_fn(**cfg.base_model),
        seed=cfg.sampling.seed,
    )

    eval_data = step1.pipe(
        agent=task_maker,
        embed_config=cfg.embed_config,
        trie=xml_trie,
        dataset=dataset,
        qdrant_service=qdrant_service,
        rank=cfg.retrieval.topk,
        num_workers=cfg.runtime.num_workers,
        batch_size=cfg.runtime.batch_size,
        seed=cfg.sampling.seed,
        hnsw=cfg.retrieval.hnsw,
        distance=cfg.retrieval.distance,
        all_codes=cfg.retrieval.all_codes,
    )

    eval_data.save_to_disk(args.output)

    exp_utils.evaluate_and_dump_metrics(
        eval_data=eval_data,
        trie=eval_trie,
        dump_path=metrics_dir,
        file_prefix="analyze",
    )


if __name__ == "__main__":
    main()
