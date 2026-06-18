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

# --- workload: vLLM concurrency test with Qwen3-0.6B ---
module load apptainer

export HF_HOME="/scratch/gp50/zs5704/hf"
SIF="/scratch/gp50/zs5704/apptainer/vllm-020.sif"
PORT=8000
MODEL="openai/gpt-oss-20b"

# Start vLLM server in background
apptainer exec --nv \
  --bind /scratch/gp50:/scratch/gp50 \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_OFFLINE=1 \
  --env TORCHINDUCTOR_CACHE_DIR="${PBS_JOBFS}/torchinductor" \
  --env TIKTOKEN_RS_CACHE_DIR=/scratch/gp50/zs5704/.cache/tiktoken-rs-cache \
  --env VLLM_CACHE_ROOT="${PBS_JOBFS}/vllm_cache" \
  "$SIF" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.70 \
    --tensor-parallel-size 1 &

VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
until curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1; do
  sleep 15
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FATAL: vLLM server died"
    exit 1
  fi
done
echo "vLLM server ready"

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="-"

REASONING_EFFORT="low"
TEST_DATA="/scratch/gp50/zs5704/projects/"

echo "[test] === model=$MODEL === with reasonging=$REASONING_EFFORT ==="

dvc exp run -S model="$MODEL" \
  -S reasoning_effort="$REASONING_EFFORT" \
  -S max_concurrent=96 \
  -S data_dir="$TEST_DATA" \
  -S code_description_path="/scratch/gp50/zs5704/datasets/mimic-data/mimic-iv-2.2/hosp/" \
  -S guidelines_path="/scratch/gp50/zs5704/projects/prompt-icd/datasets/guidelines/icd10_guidelines.parquet" \
  -S code_type="ICD-10" \
  -S search.mrconso_path="/scratch/gp50/zs5704/datasets/umls/MRCONSO.RRF" \
  -S evaluation.label_path="/scratch/gp50/zs5704/projects/prompt-icd/datasets/processed_mimic_iv_full.parquet" \
  -S evaluation.label_config.diag_column="diagnoses_code" \
  -S evaluation.label_config.proc_column="procedure_code" 


kill $VLLM_PID 2>/dev/null
echo "[run] done"