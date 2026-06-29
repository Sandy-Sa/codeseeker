from functools import partial
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError
from agents.parsing import parse_id_answer


class AssignAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices."""
        output = parse_id_answer(content)
        # An <answer> block with no IDs (e.g. "None", "N/A") yields [] -- a valid
        # "no applicable code" prediction (the retrieved candidate set can
        # genuinely contain zero correct codes), not a parse failure. Only a
        # truly missing answer (None => truncated/malformed) is retried.
        if output is None:
            raise StructuredError(
                f"Could not find any relevant answer in the response: {content[-250:]}"
            )
        return {"reasoning": content, "output": output}


class StructuredAssignAgent(AssignAgent):
    """A structured assign agent that simulates the candidate space."""

    CODE_LIST = r"(?:[1-9]\d{0,3})(?:,(?:[1-9]\d{0,3})){0,19}\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_params["guided_regex"] = self.CODE_LIST

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices into a single."""
        # match comma separated list of integers with regex
        output = [int(num.strip()) for num in content.split(",")]

        return {"reasoning": "", "output": output}


def create_assign_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "structured":
        return partial(
            StructuredAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )

    elif agent_type == "reasoning":
        return partial(
            AssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
