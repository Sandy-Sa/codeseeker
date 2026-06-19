"""DVC stage: stage a dataset's machine-local raw files into the repo tree.

Dataset-agnostic: reads the ``staging[<dataset>]`` manifest from ``params.yaml``
and copies each ``copies[*].src`` (relative to that dataset's ``raw_dir``) to its
repo-relative ``dest``. No dataset name is hard-coded here -- swapping or adding a
dataset is purely a ``params.yaml`` edit.

The raw files are distributed pre-processed (the same files ``prepare_<dataset>.py``
would emit), so "staging" is just a copy from the machine-local source location to
the path the HuggingFace generator expects.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

from stages import _config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage a dataset's raw files")
    parser.add_argument("--dataset", required=True, help="Key in params.yaml -> staging")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    args = parser.parse_args()

    cfg = _config.load_config(args.params)

    if args.dataset not in cfg.staging:
        raise KeyError(
            f"No staging manifest for `{args.dataset}`. "
            f"Available: {sorted(cfg.staging)}"
        )

    entry = cfg.staging[args.dataset]
    raw_dir = pathlib.Path(entry.raw_dir).expanduser()

    if not entry.copies:
        raise ValueError(f"staging[{args.dataset}].copies is empty -- nothing to stage.")

    for copy in entry.copies:
        src = raw_dir / copy.src
        dest = _config.REPO_ROOT / copy.dest
        if not src.exists():
            raise FileNotFoundError(
                f"Required source for `{args.dataset}` not found: {src}. "
                f"Set staging.{args.dataset}.raw_dir in params.yaml or run the "
                f"dataset's prepare_*.py."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"[stage-data:{args.dataset}] {src} -> {dest}")


if __name__ == "__main__":
    main()
