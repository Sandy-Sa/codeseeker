import hashlib
import os
import pathlib
import typing as typ
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


# Where per-stage metrics / responses are dumped. Override with $DUMP_FOLDER so
# nothing leaks into $HOME on Gadi (point it at scratch in the run script).
DUMP_FOLDER = pathlib.Path(
    os.environ.get("DUMP_FOLDER", "~/research/codeseeker/experiments")
).expanduser()


class BaseArguments(BaseSettings):
    """Args for the script."""

    experiment_id: str
    experiment_name: str

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str
    deployment: str
    pretrained_model_path: str | None = None
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"
    use_cache: bool = True  # whether to cache on request level

    dataset: str
    seed: str

    prompt_name: str

    num_workers: int
    batch_size: int

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)

    @pydantic.computed_field
    def _deployment_name(self) -> str:
        """Get the model name."""
        return self.deployment.split("/")[-1]

    @pydantic.computed_field
    def _hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(
            exclude={
                "_hash",
                "_deployment_name",
                "experiment_folder",
                "experiment_name",
                "seed",
            }
        )
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self.experiment_id}/{self.experiment_name}/{self._deployment_name}"

    @pydantic.computed_field
    def _seeds(self) -> list[int]:
        """Get the seeds."""
        return (
            [int(x) for x in self.seed.split(":")]
            if ":" in self.seed
            else [int(self.seed)]
        )

    @pydantic.computed_field
    def _datasets(self) -> list[str]:
        """Get the datasets."""
        return (
            [x for x in self.dataset.split(":")]
            if ":" in self.dataset
            else [self.dataset]
        )

    @pydantic.computed_field
    def _prompts(self) -> list[str]:
        """Get the prompts."""
        return (
            [x for x in self.prompt_name.split(":")]
            if ":" in self.prompt_name
            else [self.prompt_name]
        )
