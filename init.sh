#!/usr/bin/env bash
# Build a uv-based venv and pack it to $ENV_CACHE/<env_name>.tar.gz
# Must be run via qsub (`puv env create` does this).
# Requires: ENV_CACHE, ENV_NAME

#PBS -P gp50
#PBS -q copyq
#PBS -l ncpus=1,mem=4GB,walltime=02:00:00
#PBS -l storage=scratch/gp50
#PBS -l jobfs=50GB
#PBS -l wd

set -euo pipefail

: "${ENV_CACHE:?ENV_CACHE must be set (e.g., /scratch/gp50/zs5704/envs)}"
: "${ENV_NAME:?ENV_NAME must be set (e.g., my_env)}"
: "${MODEL_REQUIREMENTS:=model-requirements.txt}"

if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[env] ERROR: Invalid ENV_NAME '$ENV_NAME'. Use only letters, numbers, dots, underscores, or hyphens."
  exit 1
fi

# Strict: must have node-local space and uv available
[[ -n "${PBS_JOBFS:-}" ]] || { echo "[env] ERROR: PBS_JOBFS is not set."; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "[env] ERROR: 'uv' not found on PATH."; exit 127; }

mkdir -p "$ENV_CACHE"

TARBALL="$ENV_CACHE/${ENV_NAME}.tar.gz"
echo "[env] env_name=$ENV_NAME"
echo "[env] tarball=$TARBALL"

# Idempotent: skip build if tarball exists
if [[ -f "$TARBALL" ]]; then
  echo "[env] cache hit — nothing to build."
  exit 0
fi

WORKDIR="$PBS_JOBFS/work"
mkdir -p "$WORKDIR"

echo "[env] staging sources to $WORKDIR"
rsync -a --delete \
  --exclude '.git' --exclude '.venv' \
  --exclude '__pycache__' --exclude '.mypy_cache' \
  --exclude '.ruff_cache' --exclude '.pytest_cache' \
  "$PWD"/ "$WORKDIR"/

cd "$WORKDIR"

# Keep uv’s cache on jobfs
export UV_CACHE_DIR="$PBS_JOBFS/uvcache"
mkdir -p "$UV_CACHE_DIR"

echo "[env] creating venv + syncing project dependencies"
# uv venv .venv
# uv sync


TORCH_BACKEND=cu129
# echo "[env] creating venv"
uv venv --python 3.12
source .venv/bin/activate

# uv pip install tensorboard "vllm==0.11.2" torch --torch-backend=$TORCH_BACKEND
uv pip install -r requirements.txt --torch-backend=$TORCH_BACKEND


# Optionally prefetch Hugging Face models listed in $MODEL_REQUIREMENTS
if [[ -f "$MODEL_REQUIREMENTS" ]]; then
  if [[ -z "${HF_HOME:-}" ]]; then
    echo "[env] ERROR: $MODEL_REQUIREMENTS found but HF_HOME is not set."
    echo "[env]        Set HF_HOME to a writable directory (e.g., /scratch/<proj>/<user>/hf-home)."
    exit 3
  fi

  echo "[env] Hugging Face prefetch enabled"
  echo "[env] MODEL_REQUIREMENTS=$MODEL_REQUIREMENTS"
  echo "[env] HF_HOME=$HF_HOME"

  mkdir -p "$HF_HOME"
  export HF_HOME="$HF_HOME"
  export HF_HUB_ENABLE_HF_TRANSFER=1

  # Determine how to run huggingface-cli ephemerally (does not mutate the venv)
  if command -v uvx >/dev/null 2>&1; then
    HF_CLI=(uvx --from huggingface_hub[hf_transfer] hf)
  else
    HF_CLI=(uv tool run --from huggingface_hub[hf_transfer] hf)
  fi

  # Parse non-empty, non-comment lines. Supported formats:
  #   - "repo_id" (assumes model repo)
  #   - "model:repo_id" or "dataset:repo_id"
  #   - "repo_id@revision" or "model:repo_id@revision"
  mapfile -t _HF_SPECS < <(sed 's/#.*$//' "$MODEL_REQUIREMENTS" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF>0')
  if [[ ${#_HF_SPECS[@]} -gt 0 ]]; then
    echo "[env] Will prefetch ${#_HF_SPECS[@]} Hugging Face repos"
  fi

  for spec in "${_HF_SPECS[@]}"; do
    repo_type="model"
    repo="$spec"
    # repo type prefix
    if [[ "$repo" == model:* ]]; then
      repo_type="model"; repo="${repo#model:}"
    elif [[ "$repo" == dataset:* ]]; then
      repo_type="dataset"; repo="${repo#dataset:}"
    elif [[ "$repo" == space:* ]]; then
      repo_type="space"; repo="${repo#space:}"
    fi

    revision=""
    if [[ "$repo" == *@* ]]; then
      revision="${repo#*@}"
      repo="${repo%@*}"
    fi

    echo "[env] [hf] downloading $repo_type:$repo${revision:+@$revision}"
    if [[ -n "$revision" ]]; then
      "${HF_CLI[@]}" download "$repo" --repo-type "$repo_type" --revision "$revision"
    else
      "${HF_CLI[@]}" download "$repo" --repo-type "$repo_type"
    fi
  done
fi

ACTIVATE_FILE="$WORKDIR/.venv/bin/activate"
if [[ -f "$ACTIVATE_FILE" ]]; then
  echo "[env] patching activate to compute VIRTUAL_ENV from its location"
  awk '
    BEGIN{replaced=0}
    {
      if ($0 ~ /^VIRTUAL_ENV=/ && replaced==0) {
        print "VIRTUAL_ENV=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"";
        replaced=1
      } else {
        print
      }
    }
  ' "$ACTIVATE_FILE" > "$ACTIVATE_FILE.tmp" && mv "$ACTIVATE_FILE.tmp" "$ACTIVATE_FILE"
fi

# Patch console-script shebangs (including llamafactory-cli) to /usr/bin/env python3
echo "[env] patching console script shebangs to /usr/bin/env python3"
while IFS= read -r -d '' f; do
  if head -n1 "$f" | grep -Eq '^#!.*python'; then
    { echo '#!/usr/bin/env python3'; tail -n +2 "$f"; } > "$f.tmp" && mv "$f.tmp" "$f"
    chmod +x "$f"
  fi
done < <(find .venv/bin -maxdepth 1 -type f -print0)

echo "[env] packing venv to $TARBALL"
tar -C "$WORKDIR" -czf "$TARBALL" .venv
echo "[env] done."