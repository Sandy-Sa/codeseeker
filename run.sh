#!/bin/bash
#PBS -P gp50
#PBS -q gpuhopper
#PBS -l ncpus=12,mem=256g,ngpus=1,walltime=12:00:00
#PBS -l storage=scratch/gp50
#PBS -l jobfs=200GB
#PBS -l wd

set -euo pipefail

module load python3/3.12.1
module load cuda/12.9.0

# Hard requirements
[[ -n "${PBS_JOBFS:-}" ]] || { echo "[run] ERROR: PBS_JOBFS is not set."; exit 1; }
: "${TARBALL:?TARBALL not provided (`puv submit` sets this via -v)}"

if ! git -C "$PWD" rev-parse --is-inside-work-tree &>/dev/null; then
  echo "[git] ERROR: job is not running inside a git repo"
  exit 1
fi

GIT_COMMIT="$(git -C "$PWD" rev-parse HEAD)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_ENABLE_TELEMETRY=0
export LOGDIR="/scratch/gp50/zs5704/projects/codeseeker/logs"
export TIKTOKEN_CACHE_DIR=/scratch/gp50/zs5704/.cache/tiktoken

mkdir -p $LOGDIR

echo "[run] node: $(hostname)"
echo "[run] jobfs: $PBS_JOBFS"
echo "[run] tarball: $TARBALL"
echo "[run] submit cwd: $PWD"
echo "[run] target commit: $GIT_COMMIT"

WORKDIR="$PBS_JOBFS/work"

echo "[env] cloning repo from \$PWD to $WORKDIR"
git clone "$PWD" "$WORKDIR"

cd "$WORKDIR"

echo "[git] checking out commit $GIT_COMMIT"
git checkout "$GIT_COMMIT"

# Double-check we’re on the expected commit
current_commit="$(git rev-parse HEAD)"
if [[ "$current_commit" != "$GIT_COMMIT" ]]; then
  echo "[git][ERROR] expected $GIT_COMMIT but got $current_commit"
  exit 1
fi
echo "[git] on expected commit $current_commit"

# Unpack venv to node-local jobfs and activate
tar -xzf "$TARBALL" -C "$PBS_JOBFS"
source "$PBS_JOBFS/.venv/bin/activate"

# Optional sanity check (only uv/env allowed; no modules)
python -c 'import sys; print("[run] python:", sys.executable)'
python -c 'import torch; print("[run] torch.cuda.is_available:", torch.cuda.is_available())'

# --- workload: DVC agentic-coding pipeline (vLLM up -> dvc repro -> down) ------
# scripts/run_pipeline.sh launches vLLM (via the apptainer SIF below), waits for
# /v1/models, runs `dvc repro`, then tears vLLM down. The vLLM model MUST match
# `model.deployment` in params.yaml and PORT must match `model.api_base`.

export SCRATCH_BASE="/scratch/gp50/zs5704/projects/codeseeker"
export HF_HOME="/scratch/gp50/zs5704/hf"
export VLLM_SIF="/scratch/gp50/zs5704/apptainer/vllm-020.sif"
export PORT=8000
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
export TP=1
export GPU_MEM_UTIL=0.70
export QDRANT_LOCAL_PATH="${PBS_JOBFS}/.qdrant_local"
export DVC_TARGET="${DVC_TARGET:-analyse@mdace-icd10cm}"

bash "$WORKDIR/scripts/run_pipeline.sh"

echo "[run] done"