from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

T = TypeVar("T", bound=BaseModel)


async def astructured_output(
    llm: BaseChatModel,
    schema: Type[T],
    messages: list[BaseMessage],
    max_retries: int = 2,
) -> T:
    """Provider-agnostic structured output with JSON fallback on parse failure."""
    structured_llm = llm.with_structured_output(schema)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await structured_llm.ainvoke(messages)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            # Fallback: inject schema description and ask for raw JSON
            schema_desc = json.dumps(schema.model_json_schema(), indent=2)
            fallback_messages = list(messages) + [
                HumanMessage(
                    content=(
                        f"Your previous response could not be parsed. "
                        f"Respond with only valid JSON matching this schema:\n{schema_desc}"
                    )
                )
            ]
            try:
                raw = await llm.ainvoke(fallback_messages)
                text = raw.content
                # Strip markdown code fences if present
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return schema.model_validate_json(text.strip())
            except Exception:
                continue

    raise RuntimeError(
        f"astructured_output failed after {max_retries + 1} attempts "
        f"for schema {schema.__name__}"
    ) from last_exc
