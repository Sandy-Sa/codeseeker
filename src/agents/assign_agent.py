from functools import partial
import re
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError

ANSWER_PATTERN = r"<answer>.*?(\b[0-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"
ANSWER_BLOCK = r"<answer>(.*?)</answer>"


class AssignAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices."""
        content = content.replace("IDs:", "").replace("ID:", "")
        answer_match = re.search(ANSWER_PATTERN, content, re.DOTALL)
        if answer_match:
            output = [int(num.strip()) for num in answer_match.group(1).split(",")]
            return {"reasoning": content, "output": output}
        # An <answer> block with no IDs (e.g. "None", "N/A") is a valid
        # "no applicable code" prediction, not a parse failure -- the retrieved
        # candidate set can genuinely contain zero correct codes. Return an empty
        # output instead of raising, so a single such example doesn't retry 10x
        # and then abort the whole stage.
        if re.search(ANSWER_BLOCK, content, re.DOTALL):
            return {"reasoning": content, "output": []}
        # No answer block at all => truncated/malformed; let throughster retry.
        raise StructuredError(
            f"Could not find any relevant answer in the response: {content[-250:]}"
        )


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
