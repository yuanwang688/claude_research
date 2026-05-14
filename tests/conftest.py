"""Shared test fixtures."""
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from deep_research.providers.mock import MockSearchProvider


def _make_structured_output_chain(schema):
    """Return a Runnable that yields a minimal valid schema instance without LLM calls."""
    import types
    from typing import Literal, get_args, get_origin

    from pydantic.fields import PydanticUndefined

    def _default_for_annotation(ann):
        origin = get_origin(ann)
        # Literal["a", "b", ...] → use first value
        if origin is Literal:
            return get_args(ann)[0]
        # X | None or Optional[X] → use default for X
        if origin is types.UnionType or str(origin) == "typing.Union":
            args = [a for a in get_args(ann) if a is not type(None)]
            if args:
                return _default_for_annotation(args[0])
            return None
        if ann is str:
            return ""
        if ann is bool:
            return True
        if ann is float:
            return 0.5
        if ann is int:
            return 0
        if origin is list:
            return []
        if origin is dict:
            return {}
        return None

    field_values = {}
    for fname, finfo in schema.model_fields.items():
        if finfo.default is not PydanticUndefined:
            field_values[fname] = finfo.default
        elif finfo.default_factory is not None:
            field_values[fname] = finfo.default_factory()
        else:
            field_values[fname] = _default_for_annotation(finfo.annotation)

    instance = schema.model_validate(field_values)

    async def afunc(_input, **kwargs):
        return instance

    return RunnableLambda(func=lambda _: instance, afunc=afunc)


class MockChatModel(BaseChatModel):
    """Minimal mock LLM for unit tests.

    Supports with_structured_output() by returning a type-inferred minimal instance,
    so tests that run the full graph don't need to patch astructured_output.
    """

    responses: list[str] = ["mock response"]
    _call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self._call_count % len(self.responses)]
        object.__setattr__(self, "_call_count", self._call_count + 1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return _make_structured_output_chain(schema)


@pytest.fixture
def mock_llm():
    return MockChatModel()


@pytest.fixture
def mock_search():
    return MockSearchProvider()
