"""DVC stage: prepare a dataset for the agentic chain.

Loads a HuggingFace dataset via `DATASET_CONFIGS`, filters targets against the
ICD trie (`format_dataset`), applies per-dataset quirks, and serializes the full
dataset to disk so the `analyse` stage can pick it up with `load_from_disk`.
"""

from __future__ import annotations

import argparse

import dataloader
from dataloader.base import DatasetConfig
from dataloader.interface import load_dataset

import utils as exp_utils
from stages import _config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep stage")
    parser.add_argument("--dataset", required=True, help="Key in DATASET_CONFIGS")
    parser.add_argument("--output", required=True, help="save_to_disk target dir")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    args = parser.parse_args()

    cfg = _config.load_config(args.params)

    if args.dataset not in dataloader.DATASET_CONFIGS:
        raise KeyError(
            f"Unknown dataset `{args.dataset}`. "
            f"Available: {sorted(dataloader.DATASET_CONFIGS)}"
        )

    xml_trie = exp_utils.build_icd_trie(year=cfg.icd.year)
    dataset = load_dataset(DatasetConfig(**dataloader.DATASET_CONFIGS[args.dataset]))
    dataset = exp_utils.format_dataset(dataset, xml_trie, debug=False)

    staging = cfg.staging.get(args.dataset)
    skip = set(staging.skip_indices if staging else [])
    if skip:
        # Pass a concrete list (not a generator): HF can't hash a generator, so
        # it assigns a *random* dataset fingerprint, which lands in state.json and
        # makes save_to_disk non-deterministic -- DVC then sees the `prep` output
        # as modified on every run and needlessly re-runs `analyse`.
        keep = [i for i in range(len(dataset)) if i not in skip]
        dataset = dataset.select(keep)

    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
