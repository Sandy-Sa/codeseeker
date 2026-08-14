"""Per-split metrics, reported alongside the upstream all-splits numbers.

`experiments/utils.evaluate_and_dump_metrics` scores whatever it is handed, and
the released benchmark hands it *every* MDACE note: `DATASET_CONFIGS`'s
`mdace-icd10cm` entry has no `split` key, so `load_dataset` concatenates
train+val+test (550 notes after `skip_indices`). The paper, by contrast,
evaluates on the MDACE **test** split. Neither number is wrong, but only the
test one is comparable to the published tables, so we emit both.

This module deliberately does NOT call `evaluate_and_dump_metrics` a second
time: that function also writes a fixed `responses.json`, and a second call
would silently overwrite the full-set dump with a subset. It reuses the same
`TrieClassificationMonitor`, so the per-split figures are computed by exactly
the same code as the headline ones.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import pathlib
import typing as typ

import datasets
from loguru import logger
import polars as pl
import rich
import torch

from utils import TrieClassificationMonitor

# `aid` is "{hadm_id}_{note_id}"; the split file is keyed by hadm_id.
SPLIT_ID_COLUMN = "_id"


def load_split_map(splits_file: str | pathlib.Path) -> dict[int, str]:
    """hadm_id -> {train,val,test} from the consolidated split feather."""
    table = pl.read_ipc(splits_file, memory_map=False)
    return dict(
        zip(
            table[SPLIT_ID_COLUMN].cast(pl.Int64).to_list(),
            table["split"].to_list(),
        )
    )


def split_of(aid: str, split_map: dict[int, str]) -> str:
    """Split label for one `aid`, or ``"unknown"`` when the id is absent."""
    try:
        hadm_id = int(str(aid).split("_")[0])
    except (TypeError, ValueError):
        return "unknown"
    return split_map.get(hadm_id, "unknown")


def _score(
    eval_data: datasets.Dataset, trie: OrderedDict[str, int]
) -> dict[str, dict[str, float]]:
    overall = TrieClassificationMonitor(trie=trie)
    contextual = TrieClassificationMonitor(trie=trie)
    overall.update(
        target_inputs=eval_data["targets"], pred_inputs=eval_data["output"]
    )
    contextual.update(
        target_inputs=eval_data["subset_targets"], pred_inputs=eval_data["output"]
    )
    return {
        name: {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in monitor.get().items()
        }
        for name, monitor in (("overall", overall), ("contextual", contextual))
    }


def dump_split_metrics(
    *,
    eval_data: datasets.Dataset,
    trie: OrderedDict[str, int],
    dump_path: pathlib.Path,
    file_prefix: str,
    splits_file: str | pathlib.Path | None,
    splits: typ.Iterable[str] = ("test",),
) -> dict[str, dict[str, dict[str, float]]]:
    """Score `eval_data` restricted to each named split and dump the result.

    Writes ``{file_prefix}_metrics_by_split.json``. A missing `splits_file` or
    an `aid` column that does not join is logged and skipped -- per-split
    reporting must never break a stage that already produced its metrics.
    """
    if not splits_file:
        logger.warning("No splits_file configured; skipping per-split metrics.")
        return {}
    if "aid" not in eval_data.column_names:
        logger.warning("No `aid` column; skipping per-split metrics.")
        return {}

    split_map = load_split_map(splits_file)
    labels = [split_of(aid, split_map) for aid in eval_data["aid"]]
    if all(label == "unknown" for label in labels):
        logger.warning(
            f"No `aid` joined against {splits_file}; skipping per-split metrics."
        )
        return {}

    results: dict[str, dict[str, dict[str, float]]] = {}
    for split in splits:
        idx = [i for i, label in enumerate(labels) if label == split]
        if not idx:
            logger.warning(f"Split `{split}` matched no rows; skipping.")
            continue
        results[split] = _score(eval_data.select(idx), trie)
        rich.print(
            f"[blue][{file_prefix}][{split} split, n={len(idx)}] "
            f"{results[split]['overall']}[/blue]"
        )

    with open(dump_path / f"{file_prefix}_metrics_by_split.json", "w") as f:
        json.dump(results, f)

    return results
