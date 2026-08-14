"""Offline tests for per-split metric reporting (no GPU / no vLLM).

Run:
    PYTHONPATH=src:experiments python -m pytest tests/stages/test_split_metrics.py -q
"""

from collections import OrderedDict
import json

import datasets
import polars as pl
import pytest

from stages import _metrics


@pytest.fixture
def splits_file(tmp_path):
    path = tmp_path / "split.feather"
    pl.DataFrame(
        {"_id": [100, 200, 300], "split": ["train", "test", "test"]}
    ).write_ipc(path)
    return path


@pytest.fixture
def eval_data():
    return datasets.Dataset.from_dict(
        {
            # aid is "{hadm_id}_{note_id}"
            "aid": ["100_1", "200_2", "300_3"],
            "targets": [["A00"], ["B00"], ["C00"]],
            "output": [["A00"], ["B00"], ["Z99"]],
            "subset_targets": [["A00"], ["B00"], ["C00"]],
        }
    )


TRIE = OrderedDict({c: i for i, c in enumerate(["A00", "B00", "C00", "Z99"], start=1)})


def test_split_of_parses_hadm_id():
    split_map = {100: "train", 200: "test"}
    assert _metrics.split_of("100_1", split_map) == "train"
    assert _metrics.split_of("200_9999", split_map) == "test"
    assert _metrics.split_of("999_1", split_map) == "unknown"
    assert _metrics.split_of(None, split_map) == "unknown"
    assert _metrics.split_of("not-an-id", split_map) == "unknown"


def test_scores_only_the_requested_split(tmp_path, splits_file, eval_data):
    out = _metrics.dump_split_metrics(
        eval_data=eval_data,
        trie=TRIE,
        dump_path=tmp_path,
        file_prefix="assign",
        splits_file=splits_file,
    )
    # test split = rows 200_2 (correct) and 300_3 (wrong) -> recall 1/2
    assert out["test"]["overall"]["recall"] == pytest.approx(0.5, abs=1e-6)
    assert out["test"]["overall"]["precision"] == pytest.approx(0.5, abs=1e-6)
    # train row (a correct one) must not leak in and inflate it
    assert "train" not in out

    written = json.loads((tmp_path / "assign_metrics_by_split.json").read_text())
    assert written["test"]["overall"]["recall"] == pytest.approx(0.5, abs=1e-6)


def test_missing_splits_file_is_skipped_not_fatal(tmp_path, eval_data):
    assert (
        _metrics.dump_split_metrics(
            eval_data=eval_data,
            trie=TRIE,
            dump_path=tmp_path,
            file_prefix="assign",
            splits_file=None,
        )
        == {}
    )


def test_non_joining_aids_are_skipped_not_fatal(tmp_path, splits_file):
    orphan = datasets.Dataset.from_dict(
        {
            "aid": ["777_1"],
            "targets": [["A00"]],
            "output": [["A00"]],
            "subset_targets": [["A00"]],
        }
    )
    assert (
        _metrics.dump_split_metrics(
            eval_data=orphan,
            trie=TRIE,
            dump_path=tmp_path,
            file_prefix="assign",
            splits_file=splits_file,
        )
        == {}
    )
