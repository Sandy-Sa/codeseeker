from abc import abstractmethod
from collections import defaultdict
import json
from pathlib import Path
import re
import typing as typ


def _flatten_exc(exc: BaseException) -> str:
    """Render an exception (recursively unwrapping ExceptionGroups) to a string.

    throughster runs each request in an anyio task group; a non-StructuredResponseError
    (e.g. httpx.ReadTimeout after retries) escapes as an unhandled ExceptionGroup whose
    str() is only "unhandled errors in a TaskGroup (N sub-exception)". Unwrap it so the
    actual cause is visible in the log instead of being swallowed.
    """
    if isinstance(exc, BaseExceptionGroup):
        inner = "; ".join(_flatten_exc(e) for e in exc.exceptions)
        return f"{type(exc).__name__}[{inner}]"
    return f"{type(exc).__name__}: {exc}"

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import numpy as np
from prompt_poet import Prompt

from throughster.base import ModelInterface, BaseResponse
from throughster.hf_datasets import HfOperation


PATH_TO_TEMPLATES = Path(__file__).parent / "templates"


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    def sanitize_value(val):
        # Recursively sanitize strings within nested structures
        if isinstance(val, str):
            # Replace non-printable characters with a space
            return re.sub(r"[^\x20-\x7E]", " ", val)
        return val

    sanitized_value = sanitize_value(value)
    return json.dumps(sanitized_value, ensure_ascii=False)


def list2matrix(
    dim_x: int, dim_y: int, alignment_indices: list[list[int | float]]
) -> np.ndarray:
    sparse_matrix = np.zeros((dim_x, dim_y), dtype=np.float32)
    for i, preds in enumerate(alignment_indices):
        for pred in preds:
            pred_sign = -1 if pred < 0 else 1
            pred_idx = abs(pred) - 1
            if 0 <= pred_idx < dim_y:
                sparse_matrix[i, pred_idx] = pred_sign
    return sparse_matrix


# @numba.jit(cache=True, nogil=True, fastmath=True)
def matrix2list(sparse_matrix: np.ndarray) -> list:
    alignment_indices = []
    for i in range(sparse_matrix.shape[0]):
        non_zero_indices = np.where(sparse_matrix[i] == 1)[0] + 1
        if len(non_zero_indices) > 0:
            row_indices = [non_zero_indices[j] for j in range(len(non_zero_indices))]
            alignment_indices.append(row_indices)
        else:
            alignment_indices.append([0])  # Append [0] for zero rows
    return alignment_indices


class HfBaseAgent(HfOperation):
    """Base class for coding agents."""

    def __init__(
        self,
        init_client_fn: typ.Callable[..., ModelInterface],
        prompt_name: str,
        seed: int,
        sampling_params: dict[str, typ.Any],
        max_retries: int = 10,
    ):
        self.init_client_fn = init_client_fn
        env = Environment(loader=FileSystemLoader(PATH_TO_TEMPLATES), autoescape=False)
        loader = typ.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(
            env, f"{prompt_name}.yml.j2"
        )
        self.prompt_name = prompt_name
        self.seed = seed
        self.sampling_params = sampling_params
        self.max_retries = max_retries
        self._client: ModelInterface | None = None

    @property
    def client(self) -> ModelInterface:
        if self._client is None:
            self._client = self.init_client_fn()
        return self._client

    def format_request(self, **kwargs) -> dict[str, typ.Any]:
        """Format the prompt."""
        prompt_template = Prompt(
            raw_template=self.raw_template,
            template_data={"custom_tojson": custom_tojson, **kwargs},
        )
        prompt = self.prompt_messages_or_string(self.client, prompt_template)
        return {
            "prompt": prompt if self.client.endpoint == "completions" else None,
            "messages": prompt if self.client.endpoint == "chat/completions" else None,
            "seed": self.seed,
            **self.sampling_params,
        }

    # Every agent template ends its user turn with a literal `<think>` so that
    # the raw `completions` path primes DeepSeek-R1's reasoning by hand. On
    # `chat/completions` vLLM applies the model's own chat template, which
    # already appends `<|Assistant|><think>\n`; leaving ours in would emit a
    # stray `<think>` *inside* the user turn, before the assistant header.
    _TRAILING_THINK = re.compile(r"\s*<think>\s*\Z")

    @classmethod
    def prompt_messages_or_string(
        cls, client: ModelInterface, prompt: Prompt
    ) -> str | list[dict[str, str]]:
        if client.endpoint == "chat/completions":
            messages = [dict(m) for m in prompt.messages]
            if messages:
                messages[-1]["content"] = cls._TRAILING_THINK.sub(
                    "", messages[-1]["content"]
                )
            return messages
        return prompt.string

    def __call__(
        self, batch: dict[str, list[typ.Any]], *args, **kwargs
    ) -> dict[str, list[typ.Any]]:
        """Process a row of agent tasks from a HuggingFace datasets.map()."""
        batch_size = len(batch[list(batch.keys())[0]])
        try:
            batch_rows: list[dict[str, typ.Any]] = [
                self.format_request(**{key: value[i] for key, value in batch.items()})
                for i in range(batch_size)
            ]
        except Exception:
            logger.warning(
                f"Failed to format request for batch: {batch}. "
                "Please check the input data and the format_request method."
            )

        try:
            responses = self.batch_call(batch_rows)
        except Exception as exc:
            # batch_call may *raise* (not just return) a client error such as
            # httpx.HTTPStatusError, which cannot be reconstructed by dill when
            # it crosses the datasets.map(num_proc>1) worker boundary -- that
            # silently deadlocks the pool. Convert to a picklable RuntimeError so
            # the stage fails loudly instead of hanging.
            raise RuntimeError(
                f"LLM batch_call failed: {_flatten_exc(exc)}"
            ) from None

        output = defaultdict(list)
        for resp in responses:
            if isinstance(resp, Exception):
                # Re-raise as a picklable RuntimeError. Client exceptions such as
                # httpx.HTTPStatusError cannot be reconstructed by dill when
                # returned from a datasets.map(num_proc>1) worker -- that silently
                # kills the pool's results-handler thread and deadlocks the parent
                # process (it spins a CPU forever instead of failing). A plain
                # RuntimeError always pickles, so the stage aborts loudly with a
                # clear message. `from None` avoids re-chaining the unpicklable
                # original as __cause__.
                raise RuntimeError(
                    f"LLM request failed after retries: {type(resp).__name__}: {resp}"
                ) from None
            try:
                resp_dict = self.parser(resp.choices[0].content)
            except Exception:
                raise ValueError(
                    f"Failed to parse response: {resp.choices[0].content}"
                ) from None
            for key, value in resp_dict.items():
                output[key].append(value)

        return {
            **batch,
            **output,
        }

    def batch_call(self, requests: list[dict[str, typ.Any]]) -> list[BaseResponse]:
        """Async wrapper for the translation operation."""
        return self.client.sync_structured_batch_call(
            requests, self.parser, self.max_retries
        )

    @abstractmethod
    def parser(self, str) -> dict[str, typ.Any]:
        """Condition for parsing the response."""
        ...
