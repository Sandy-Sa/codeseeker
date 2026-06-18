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

# Per-dataset rows to drop after formatting (e.g. known-bad MDACE notes). These
# were hard-coded in the original `benchmark.py`; keyed by dataset here.
SKIP_INDICES: dict[str, list[int]] = {
    "mdace-icd10cm": [27, 179, 260, 327, 379, 394],
}


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

    skip = set(SKIP_INDICES.get(args.dataset, []))
    if skip:
        dataset = dataset.select(i for i in range(len(dataset)) if i not in skip)

    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
