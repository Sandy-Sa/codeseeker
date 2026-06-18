#!/bin/bash
# Offline pre-staging — run on a node WITH internet (login / copyq / data-mover).
#
# Compute nodes are offline, so everything the pipeline downloads must be fetched
# here onto scratch first:
#   1. the frozen CMS ICD XML (ICD10Trie.from_cms) -> $TRIE_CACHE_DIR
#   2. the embedding model -> $HF_HOME (vLLM model is handled by init.sh)
#
# Requires the project venv to be active (or pass the python via $PYTHON).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gp50/zs5704/projects/codeseeker}"
export HF_HOME="${HF_HOME:-/scratch/gp50/zs5704/hf}"
export TRIE_CACHE_DIR="${TRIE_CACHE_DIR:-$SCRATCH_BASE/cache/trie}"
export ICD_YEAR="${ICD_YEAR:-2022}"
PYTHON="${PYTHON:-python}"

mkdir -p "$HF_HOME" "$TRIE_CACHE_DIR"

echo "[prestage] HF_HOME=$HF_HOME"
echo "[prestage] TRIE_CACHE_DIR=$TRIE_CACHE_DIR"

# 1. Freeze the CMS ICD XML into the trie cache (downloads from cms.gov online).
echo "[prestage] downloading + parsing CMS ICD-$ICD_YEAR ..."
PYTHONPATH="src:experiments" "$PYTHON" - <<PY
import os
from trie.icd import ICD10Trie
year = int(os.environ["ICD_YEAR"])
trie = ICD10Trie.from_cms(year=year)
trie.parse()
print(f"[prestage] ICD-{year} trie ready ({len(trie.lookup)} codes)")
PY

# 2. Prefetch the embedding model (vLLM model is prefetched by init.sh).
echo "[prestage] prefetching embedding model ..."
HF_HUB_ENABLE_HF_TRANSFER=1 "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("pritamdeka/S-PubMedBert-MS-MARCO")
print("[prestage] embedding model cached")
PY

echo "[prestage] done"
