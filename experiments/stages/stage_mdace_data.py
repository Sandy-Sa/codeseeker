"""DVC stage: stage the (already processed) MDACE parquet into the repo tree.

The MDACE annotations are distributed pre-processed (the same files that
`prepare_mdace.py` would emit), so "data prep" here only copies them from the
machine-local scratch location (``raw.mdace_dir`` in params.yaml) to the path the
HuggingFace generator expects: ``data/mdace/processed/``.

If you instead have the raw MDACE JSON (Inpatient/Profee), run
``prepare_mdace.py`` to regenerate these parquets.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

from stages import _config

# Files the MDACE generator / adapter read from data/mdace/processed/.
_REQUIRED = ["mdace_inpatient_annotations.parquet"]
_OPTIONAL = ["mdace_notes.parquet"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage MDACE processed data")
    parser.add_argument(
        "--output-dir",
        default=str(_config.REPO_ROOT / "data/mdace/processed"),
        help="Destination dir (default: data/mdace/processed)",
    )
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    args = parser.parse_args()

    import yaml

    params_path = args.params or _config.DEFAULT_PARAMS_PATH
    with open(params_path) as f:
        raw_cfg = (yaml.safe_load(f) or {}).get("raw", {})
    src_dir = pathlib.Path(raw_cfg["mdace_dir"]).expanduser()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in _REQUIRED:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"Required MDACE file not found: {src}. "
                f"Set raw.mdace_dir in params.yaml or run prepare_mdace.py."
            )
        shutil.copy2(src, out_dir / name)
        print(f"[stage-mdace] {src} -> {out_dir / name}")

    for name in _OPTIONAL:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"[stage-mdace] {src} -> {out_dir / name}")


if __name__ == "__main__":
    main()
