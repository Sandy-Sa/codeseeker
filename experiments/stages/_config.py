"""Shared configuration + helpers for the DVC stage entrypoints.

Every stage (`prep`, `analyse`, `locate`, `verify`, `assign`) reads the same
`params.yaml` so that DVC can hash the parameters and resolve cache hits across
machines. Heavy artifacts (the vLLM model, the embedding model) are referenced
by name + pinned revision only -- they are never DVC deps.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import pathlib
import typing as typ

import pydantic
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = REPO_ROOT / "params.yaml"


class ModelConfig(pydantic.BaseModel):
    """LLM served by vLLM (OpenAI-compatible endpoint)."""

    provider: str = "vllm"
    api_base: str = "http://localhost:8000/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    revision: str | None = None  # pinned HF revision SHA (hashed, never downloaded)
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"
    use_cache: bool = True


class SamplingConfig(pydantic.BaseModel):
    temperature: float = 0.6
    max_tokens: int = 50_000
    seed: int = 1
    reasoning_effort: str = "low"


class RetrievalConfig(pydantic.BaseModel):
    embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embed_revision: str | None = None
    embed_query_key: str = "output"
    distance: str = "Cosine"
    topk: int = 10
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}
    all_codes: bool = False


class QdrantConfig(pydantic.BaseModel):
    """Embedded Qdrant. ``local_path`` may be overridden by $QDRANT_LOCAL_PATH."""

    local_path: str = "./.qdrant_local"


class AgentConfig(pydantic.BaseModel):
    agent_type: str
    prompt_name: str


class IcdConfig(pydantic.BaseModel):
    year: int = 2022


class RuntimeConfig(pydantic.BaseModel):
    num_workers: int = 4
    batch_size: int = 1


class PipelineConfig(pydantic.BaseModel):
    """Top-level view of ``params.yaml``."""

    datasets: list[str] = ["mdace-icd10cm"]
    model: ModelConfig = ModelConfig()
    sampling: SamplingConfig = SamplingConfig()
    icd: IcdConfig = IcdConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    qdrant: QdrantConfig = QdrantConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    agents: dict[str, AgentConfig] = {}

    # ---- derived helpers -------------------------------------------------

    @property
    def embed_config(self) -> list[dict[str, str]]:
        return [
            {
                "type": "st",
                "model_name": self.retrieval.embed_model,
                "query_key": self.retrieval.embed_query_key,
            }
        ]

    @property
    def base_model(self) -> dict[str, typ.Any]:
        """Kwargs for `utils._init_client_fn`."""
        return {
            "provider": self.model.provider,
            "api_base": self.model.api_base,
            "endpoint": self.model.endpoint,
            "deployment": self.model.deployment,
            "use_cache": self.model.use_cache,
        }

    def sampling_params(self) -> dict[str, typ.Any]:
        # NOTE: the OpenAI-compatible ``/v1/completions`` endpoint (model.endpoint
        # == "completions") honours ``max_tokens`` -- it silently ignores
        # ``max_completion_tokens`` (a chat/completions-only field) and falls back
        # to a small default, truncating the reasoning before the <answer> block.
        # Use ``max_tokens`` so the agent's structured output is actually emitted.
        return {
            "temperature": self.sampling.temperature,
            "max_tokens": self.sampling.max_tokens,
            "seed": self.sampling.seed,
            "model": self.model.deployment,
            "reasoning_effort": self.sampling.reasoning_effort,
        }

    def qdrant_local_path(self) -> str:
        return os.environ.get("QDRANT_LOCAL_PATH", self.qdrant.local_path)


def load_config(path: str | os.PathLike | None = None) -> PipelineConfig:
    """Load and validate `params.yaml`."""
    params_path = pathlib.Path(path) if path else DEFAULT_PARAMS_PATH
    with open(params_path) as f:
        raw = yaml.safe_load(f) or {}
    return PipelineConfig.model_validate(raw)


def build_eval_trie(xml_trie) -> OrderedDict[str, int]:
    """ICD-10-CM code -> index map used by the metrics monitor."""
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    return OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )
