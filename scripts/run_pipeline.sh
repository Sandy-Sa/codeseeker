#!/bin/bash
# Inner run script for the DVC agentic-coding pipeline.
#
# PBS-agnostic: run it directly on any machine that has the venv activated and a
# GPU for vLLM, or call it from a PBS wrapper (see run.sh). It:
#   1. exports offline + scratch-cache env (so nothing leaks into $HOME),
#   2. launches vLLM (OpenAI-compatible) in the background,
#   3. waits for the server to be healthy,
#   4. runs `dvc repro` for the requested target,
#   5. tears vLLM down on exit.
#
# The vLLM `--model` MUST match `model.deployment` in params.yaml, and $PORT must
# match the port in `model.api_base`, because throughster talks to that endpoint.
#
# Override any of the variables below via the environment, e.g.:
#   MODEL=openai/gpt-oss-20b DVC_TARGET=analyse@mdace-icd10cm ./scripts/run_pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- configuration (override via env) -----------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gp50/zs5704/projects/codeseeker}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                       # tensor-parallel size
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DVC_TARGET="${DVC_TARGET:-analyse@mdace-icd10cm}"

# Offline HF (compute nodes have no internet). Pre-stage models with init.sh /
# scripts/prestage.sh first.
export HF_HOME="${HF_HOME:-/scratch/gp50/zs5704/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_ENABLE_TELEMETRY=0

# Redirect incidental caches off $HOME onto scratch (speed only; not DVC deps).
export TRIE_CACHE_DIR="${TRIE_CACHE_DIR:-$SCRATCH_BASE/cache/trie}"
export DUMP_FOLDER="${DUMP_FOLDER:-$SCRATCH_BASE/dumps}"
export THROUGHSTER_CACHE_DIR="${THROUGHSTER_CACHE_DIR:-$SCRATCH_BASE/cache/throughster}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME}"
# Embedded Qdrant index on node-local fast storage if available.
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-${PBS_JOBFS:-$REPO_ROOT}/.qdrant_local}"

# throughster talks to this endpoint via the OpenAI client.
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:--}"

mkdir -p "$TRIE_CACHE_DIR" "$DUMP_FOLDER" "$THROUGHSTER_CACHE_DIR"

echo "[pipeline] repo:        $REPO_ROOT"
echo "[pipeline] model:       $MODEL  (port $PORT, tp=$TP)"
echo "[pipeline] dvc target:  $DVC_TARGET"
echo "[pipeline] qdrant path: $QDRANT_LOCAL_PATH"

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

if [[ -n "${VLLM_SIF:-}" ]]; then
  # Apptainer/Singularity image path (Gadi).
  module load apptainer 2>/dev/null || true
  apptainer exec --nv \
    --bind /scratch:/scratch \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
    --env VLLM_CACHE_ROOT="${PBS_JOBFS:-/tmp}/vllm_cache" \
    "$VLLM_SIF" python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
      --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM_UTIL" &
else
  # Native vLLM in the activated venv.
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM_UTIL" &
fi
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
dvc repro "$DVC_TARGET"

echo "[pipeline] done"
