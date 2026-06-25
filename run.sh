#!/bin/bash
# PBS job: reproduce the agentic-coding benchmark on Gadi.
#   clone repo @ pinned commit -> unpack venv -> launch vLLM (apptainer SIF) ->
#   wait for /v1/models -> `dvc repro` -> tear vLLM down.
#
# Self-contained: the vLLM launch + dvc repro logic is inlined here (the old
# scripts/run_pipeline.sh is gone). The vLLM `--model` MUST equal
# `model.deployment` in params.yaml, and $PORT MUST match `model.api_base`
# (http://localhost:$PORT/v1), because the agents talk to that endpoint.
#
# Submit with:  puv submit  (sets $TARBALL via -v), or:
#   qsub -v TARBALL=/scratch/gp50/zs5704/envs/<env>.tar.gz run.sh

#PBS -P gp50
#PBS -q gpuhopper
# 70B in bf16 (~140GB weights) does NOT fit one H200 (141GB); needs TP=2.
# gpuhopper is 48 cores / 4 GPUs / 1TiB per node -> request proportionally.
#PBS -l ncpus=24,mem=500g,ngpus=2,walltime=12:00:00
#PBS -l storage=scratch/gp50
#PBS -l jobfs=200GB
#PBS -l wd

set -euo pipefail

module load python3/3.12.1
module load cuda/12.9.0

# Hard requirements
[[ -n "${PBS_JOBFS:-}" ]] || { echo "[run] ERROR: PBS_JOBFS is not set."; exit 1; }
: "${TARBALL:?TARBALL not provided (\`puv submit\` sets this via -v)}"

if ! git -C "$PWD" rev-parse --is-inside-work-tree &>/dev/null; then
  echo "[git] ERROR: job is not running inside a git repo"
  exit 1
fi

GIT_COMMIT="$(git -C "$PWD" rev-parse HEAD)"

# Offline HF (compute nodes have no internet; models pre-staged on copyq).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_ENABLE_TELEMETRY=0
export LOGDIR="/scratch/gp50/zs5704/projects/codeseeker/logs"
export TIKTOKEN_CACHE_DIR=/scratch/gp50/zs5704/.cache/tiktoken

mkdir -p "$LOGDIR"

echo "[run] node:          $(hostname)"
echo "[run] jobfs:         $PBS_JOBFS"
echo "[run] tarball:       $TARBALL"
echo "[run] submit cwd:    $PWD"
echo "[run] target commit: $GIT_COMMIT"

WORKDIR="$PBS_JOBFS/work"

echo "[env] cloning repo from \$PWD to $WORKDIR"
git clone "$PWD" "$WORKDIR"
cd "$WORKDIR"

echo "[git] checking out commit $GIT_COMMIT"
git checkout "$GIT_COMMIT"
current_commit="$(git rev-parse HEAD)"
if [[ "$current_commit" != "$GIT_COMMIT" ]]; then
  echo "[git][ERROR] expected $GIT_COMMIT but got $current_commit"
  exit 1
fi
echo "[git] on expected commit $current_commit"

# Unpack venv to node-local jobfs and activate
tar -xzf "$TARBALL" -C "$PBS_JOBFS"
source "$PBS_JOBFS/.venv/bin/activate"

python -c 'import sys; print("[run] python:", sys.executable)'
python -c 'import torch; print("[run] torch.cuda.is_available:", torch.cuda.is_available())'

# --- pipeline configuration (override via env) --------------------------------
export SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gp50/zs5704/projects/codeseeker}"
export HF_HOME="${HF_HOME:-/scratch/gp50/zs5704/hf}"
export VLLM_SIF="${VLLM_SIF:-/scratch/gp50/zs5704/apptainer/vllm-020.sif}"
export PORT="${PORT:-8000}"
# MUST equal params.yaml model.deployment.
export MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
export TP="${TP:-2}"                      # tensor-parallel size (2 H200 for the 70B)
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Empty (default) => reproduce the entire DAG (stage_data -> prep -> analyse ->
# locate -> verify -> assign, for every dataset). Set to a stage like
# `analyse@mdace-icd10cm` to run just that stage and its upstream deps.
export DVC_TARGET="${DVC_TARGET:-}"

# Caches off $HOME onto scratch / node-local jobfs (speed only; not DVC deps).
export TRIE_CACHE_DIR="${TRIE_CACHE_DIR:-$SCRATCH_BASE/cache/trie}"
export DUMP_FOLDER="${DUMP_FOLDER:-$SCRATCH_BASE/dumps}"
export THROUGHSTER_CACHE_DIR="${THROUGHSTER_CACHE_DIR:-$SCRATCH_BASE/cache/throughster}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME}"
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$PBS_JOBFS/.qdrant_local}"
mkdir -p "$TRIE_CACHE_DIR" "$DUMP_FOLDER" "$THROUGHSTER_CACHE_DIR"

echo "[pipeline] model:       $MODEL  (port $PORT, tp=$TP, mem_util=$GPU_MEM_UTIL)"
echo "[pipeline] dvc target:  ${DVC_TARGET:-<full DAG>}"
echo "[pipeline] qdrant path: $QDRANT_LOCAL_PATH"
echo "[pipeline] vllm sif:    $VLLM_SIF"

# --- launch vLLM --------------------------------------------------------------
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[pipeline] stopping vLLM ($VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

module load apptainer 2>/dev/null || true
apptainer exec --nv \
  --bind /scratch:/scratch \
  --env HF_HOME="$HF_HOME" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env VLLM_CACHE_ROOT="$PBS_JOBFS/vllm_cache" \
  "$VLLM_SIF" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM_UTIL" &
VLLM_PID=$!

echo "[pipeline] waiting for vLLM /v1/models ..."
until curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; do
  sleep 10
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[pipeline] FATAL: vLLM server died before becoming ready"
    exit 1
  fi
done
echo "[pipeline] vLLM ready"

# --- run the pipeline ---------------------------------------------------------
# Outputs are written into the shared scratch DVC cache (.dvc/config), so they
# survive even though $WORKDIR is on jobfs.
if [[ -n "$DVC_TARGET" ]]; then
  dvc repro "$DVC_TARGET"
else
  dvc repro
fi

echo "[run] done"
