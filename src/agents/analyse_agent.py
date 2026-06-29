import ast
import json
from functools import partial
import re
import typing as typ
from loguru import logger

import pydantic
from throughster.core.models import ResponseChoice, BaseResponse

from agents.base import HfBaseAgent
from agents.errors import StructuredError
from agents.parsing import decode_byte_bpe, strip_thinking

ANSWER_PATTERN = r"(?s)</?answer>(\s*[^<]+)"
THINKING_PATTERN = r"<think>(.*?)<\/think>"
JSON_PATTERN = r"```json\s*(\[[\s\S]*?\])\s*```"
LIST_PATTERN = r"(?P<list>\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\])"
FAKE_LIST_PATTERN = (
    r"(?P<fake_block>"
    r"(?:list\s*\[\s*(?:dict\s*[\(\[].*?[\)\]]\s*,?\s*)+\s*\])"
    r"|(?:dict\s*[\(\[].*?[\)\]]\s*,?\s*){1,})"
)


class BaseAnalyseAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def _validate_input(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the input."""
        if "note" not in row:
            raise ValueError(f"Missing `note` in input data: {row}")
        return row

    def parser(self, content: str) -> dict[str, typ.Any]:

        content = decode_byte_bpe(content)
        m = re.search(ANSWER_PATTERN, content, re.DOTALL)
        if m:
            raw: str = self._normalise_raw(m.group(1))
            if not raw:
                raise StructuredError(f"Answer section is empty: {m.group(1)} ")
            terms = self._tokenise(raw)
            if not terms:
                raise StructuredError(
                    f"Could not parse the structured response: {raw}"
                )
            cleaned = {self._clean_term(t) for t in terms if t.strip()}
        else:
            # Reasoning models (DeepSeek-R1, Qwen, Gemma, ...) frequently emit the
            # final list AFTER </think> as a markdown / newline-separated list
            # WITHOUT the requested <answer> tags. Recover it instead of
            # discarding an otherwise correct response (which previously triggered
            # a 10x temperature-escalating retry storm and collapsed recall).
            # This branch is strictly additive: well-formed <answer> output above
            # is parsed exactly as before, so the original behaviour is preserved.
            body = strip_thinking(content)
            cleaned = {
                t for ln in body.splitlines() if (t := self._clean_term(ln)) and self._is_term(t)
            }

        cleaned = {t for t in cleaned if t}
        if not cleaned:
            raise StructuredError(
                f"Could not find <answer> tags or recover a term list in response: {content[-250:]}"
            )

        return {"reasoning": content, "output": list(sorted(cleaned))}

    # Leading markdown list markers: "- ", "* ", "• ", "1. ", "2) " ...
    _BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
    # Prose lead-ins / trailers the model wraps around a tag-less list, e.g.
    # "The current conditions extracted ... are:" / "These conditions are ...".
    _LEAD_INS = ("the ", "these ", "this ", "here ", "based on", "note:")

    @staticmethod
    def _clean_term(t: str) -> str:
        """Strip markdown decoration and trailing punctuation from one line."""
        t = t.replace("**", "").replace("__", "")
        t = BaseAnalyseAgent._BULLET.sub("", t)
        return t.strip().strip('"').strip("'").rstrip(".").rstrip(",").strip()

    @staticmethod
    def _is_term(s: str) -> bool:
        """Heuristic: keep short term-like lines, drop explanatory prose."""
        s = s.strip()
        if not s or len(s) > 80:  # a clinical term is short; prose is long
            return False
        if s.endswith(":"):  # lead-in like "...conditions extracted are:"
            return False
        return not s.lower().startswith(BaseAnalyseAgent._LEAD_INS)

    @staticmethod
    def _normalise_raw(raw: str) -> str:
        """Trim whitespace + the model's 'courtesy commas'."""
        return raw.strip().lstrip(",").rstrip(",").strip()

    @staticmethod
    def _tokenise(raw: str) -> list[str]:
        """Tokenise the raw string into a list of terms."""
        # Attempt JSON list recovery
        try:
            return json.loads(f"[{raw}]")
        except json.JSONDecodeError:
            pass

        # If quoted items exist
        quoted = re.findall(r'"([^"]+)"', raw)
        if quoted:
            return quoted

        # Choose separator based on presence
        if "\n" in raw:
            return [t.strip() for t in raw.splitlines() if t.strip()]
        return [t.strip() for t in raw.split(",") if t.strip()]


class InContextAgent(BaseAnalyseAgent):
    """An in-context analyse agent that choose index terms from a give list."""

    def _validate_input(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the input."""
        if "note" not in row or "terms" not in row:
            raise ValueError(f"Missing `note` or `terms` in input data: {row}")
        return row


class QueryAnalyseAgent(BaseAnalyseAgent):
    """A dummy assign agent that simulates the candidate space"""

    async def predict(self, input_data: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Handle a batch of alignment tasks."""
        request = self.format_request(**input_data)
        response: BaseResponse = await self.client.call(request=request)

        return self.compress_choices(response.choices)

    def compress_choices(self, choices: list[ResponseChoice]) -> dict[str, typ.Any]:
        """Compress the choices."""
        c = choices[0]
        response = ""
        answer_match = re.search(ANSWER_PATTERN, c.content, re.DOTALL)
        thinking_match = re.search(THINKING_PATTERN, c.content, re.DOTALL)
        if thinking_match:
            response += c.content.replace(thinking_match.group(0), "")
        json_match = re.search(JSON_PATTERN, c.content, re.DOTALL)
        list_match = re.search(LIST_PATTERN, c.content, re.DOTALL)
        fake_list_pattern = re.search(FAKE_LIST_PATTERN, c.content, re.DOTALL)
        if not (json_match or list_match or answer_match or fake_list_pattern):
            logger.warning(
                f"Could not find any relevant answer in the response: {c.content[-250:]}"
            )
            return {"reasoning": response, "output": []}
        parsed = None
        try:
            if json_match:
                parsed = json.loads(json_match.group(1).strip())  # type: ignore
            elif list_match:
                parsed = ast.literal_eval(list_match.group("list").strip())
            elif answer_match and fake_list_pattern:
                parsed = self.parse_pseudo_dicts(
                    fake_list_pattern.group("fake_block").strip()
                )
        except (SyntaxError, json.JSONDecodeError):
            pass

        if not parsed:
            logger.warning(f"Unexpected response format: {c.content}")
            return {"reasoning": response, "output": []}
        try:
            return {
                "reasoning": response,
                "output": [dict_data for dict_data in parsed],
            }
        except pydantic.ValidationError as e:
            logger.warning(f"Could not parse the response: {parsed}. Error: {e}")
            return {"reasoning": response, "output": []}

    @staticmethod
    def parse_pseudo_dicts(raw: str) -> list:
        """Convert the “pseudo-Python” list/dict syntax into a real list[dict] that json.loads can parse.

        Recognized forms
        ----------------
        * 'list[ dict( … ), dict( … ) ]'
        *  list[ dict[ … ], dict[ … ] ]
        *  dict( … ), dict( … ), …           (no outer list)
        *  dict[ … ], dict[ … ], …           (no outer list)
        """
        # strip outer quotes
        if raw and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
            raw = raw[1:-1].strip()

        # list[ / dict[ / dict( ➜ [  /  {
        raw = raw.replace("list[", "[")
        raw = raw.replace("dict[", "{").replace("dict(", "{")

        # close dict blocks written with ] or )
        raw = re.sub(r"^[ \t]+\]\s*(,?)\s*$", r"}\1", raw, flags=re.M)
        raw = re.sub(r"^[ \t]+\)\s*(,?)\s*$", r"}\1", raw, flags=re.M)
        raw = re.sub(r"\)\s*,", r"},", raw)  # dict(...) ,  → {...},
        raw = re.sub(r"\)\s*\]", r"}]", raw)  # dict(...)]   → {...}]

        # handle foo=bar ➜ "foo": bar   **only outside quotes**
        parts = re.split(r'(".*?(?<!\\)")', raw, flags=re.DOTALL)
        for i in range(0, len(parts), 2):  # even pieces are outside quotes
            parts[i] = re.sub(r"(\b\w+)\s*=", r'"\1":', parts[i])
        raw = "".join(parts)

        # quote bareword keys that start an object member
        raw = re.sub(
            r"([{\[,]\s*)(\w+)\s*:",
            lambda m: f'{m.group(1)}"{m.group(2)}":',
            raw,
        )

        # add a comma when two objects touch
        raw = re.sub(r"}\s*\{", "},{", raw)

        # remove trailing commas before } or ]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        # balance any missing ] caused by omitted opening list
        if re.match(r"\s*{", raw):  # starts with { but not with [
            raw = "[" + raw
        if raw.count("[") > raw.count("]"):
            raw += "]"
        return json.loads(raw)


def create_analyse_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "base":
        return partial(
            BaseAnalyseAgent,
            prompt_name=prompt_name,
            sampling_params=sampling_params,
        )
    elif agent_type == "query":
        return partial(
            QueryAnalyseAgent,
            prompt_name=prompt_name,
            sampling_params=sampling_params,
        )
    elif agent_type == "in_context":
        return partial(
            InContextAgent,
            prompt_name=prompt_name,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
