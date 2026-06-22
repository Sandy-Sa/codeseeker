#!/bin/bash
# PBS job: offline pre-staging on copyq (the ONLY Gadi queue with internet).
#
# Compute nodes are offline, so everything the pipeline downloads must be fetched
# here onto scratch first:
#   1. the frozen CMS ICD XML (ICD10Trie.from_cms)        -> $TRIE_CACHE_DIR
#   2. the embedding model (pinned revision)              -> $HF_HOME
# The vLLM model is prefetched separately by init.sh (model-requirements.txt).
#
# Needs the built project venv (trie/pymupdf/bs4/huggingface_hub), so it unpacks
# the env tarball like run.sh. Run init.sh first to produce $TARBALL.
#
# Submit from the repo root with:
#   qsub -v TARBALL=/scratch/gp50/zs5704/envs/<env>.tar.gz scripts/prestage.sh

#PBS -P gp50
#PBS -q copyq
# copyq caps a job at 1 CPU; it is the only queue with external network access.
#PBS -l ncpus=1,mem=32GB,walltime=02:00:00
#PBS -l storage=scratch/gp50
#PBS -l jobfs=50GB
#PBS -l wd

set -euo pipefail

# Under `-l wd` the cwd is the submit dir (repo root).
REPO_ROOT="${PBS_O_WORKDIR:-$PWD}"
cd "$REPO_ROOT"

[[ -n "${PBS_JOBFS:-}" ]] || { echo "[prestage] ERROR: PBS_JOBFS is not set."; exit 1; }
: "${TARBALL:?TARBALL not provided (run init.sh first; pass via qsub -v)}"

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gp50/zs5704/projects/codeseeker}"
export HF_HOME="${HF_HOME:-/scratch/gp50/zs5704/hf}"
export TRIE_CACHE_DIR="${TRIE_CACHE_DIR:-$SCRATCH_BASE/cache/trie}"
export ICD_YEAR="${ICD_YEAR:-2022}"
# Keep the embedding revision in lockstep with params.yaml retrieval.embed_revision.
EMBED_MODEL="${EMBED_MODEL:-pritamdeka/S-PubMedBert-MS-MARCO}"
EMBED_REVISION="${EMBED_REVISION:-96786c7024f95c5aac7f2b9a18086c7b97b23036}"

mkdir -p "$HF_HOME" "$TRIE_CACHE_DIR"

# Unpack + activate the built venv on node-local jobfs.
tar -xzf "$TARBALL" -C "$PBS_JOBFS"
source "$PBS_JOBFS/.venv/bin/activate"
PYTHON="${PYTHON:-python}"

echo "[prestage] HF_HOME=$HF_HOME"
echo "[prestage] TRIE_CACHE_DIR=$TRIE_CACHE_DIR"
echo "[prestage] embed=$EMBED_MODEL@$EMBED_REVISION"

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

# 2. Prefetch the embedding model at its pinned revision.
echo "[prestage] prefetching embedding model ..."
HF_HUB_ENABLE_HF_TRANSFER=1 EMBED_MODEL="$EMBED_MODEL" EMBED_REVISION="$EMBED_REVISION" "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["EMBED_MODEL"], revision=os.environ["EMBED_REVISION"])
print("[prestage] embedding model cached")
PY

echo "[prestage] done"
