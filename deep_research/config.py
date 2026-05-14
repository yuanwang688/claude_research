from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass
class Config:
    # Research loop
    max_research_loops: int = 3
    breadth: int = 3
    max_results_per_query: int = 5
    max_concurrent_searches: int = 5

    # Checkpoints
    enable_clarification: bool = True
    enable_gap_review: bool = True
    enable_draft_review: bool = True

    # Context management (round-based: compress findings after each loop)
    max_findings_tokens: int = 8000

    # Persistence
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    checkpointer: Any | None = None  # None = MemorySaver


@dataclass
class Configuration:
    """Runtime dependencies passed to nodes via RunnableConfig.configurable."""

    fast_llm: BaseChatModel
    powerful_llm: BaseChatModel
    search_provider: Any  # SearchProvider Protocol
    config: Config
    semaphore: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)

    @classmethod
    def from_runnable_config(cls, runnable_config: dict) -> Configuration:
        configurable = runnable_config.get("configurable", {})
        cfg = configurable.get("_configuration")
        if cfg is None:
            raise ValueError(
                "Configuration not found in runnable config. "
                "Ensure DeepResearchAgent is driving the graph."
            )
        return cfg

    # Convenience proxies so node code reads cleanly
    @property
    def max_research_loops(self) -> int:
        return self.config.max_research_loops

    @property
    def breadth(self) -> int:
        return self.config.breadth

    @property
    def max_results_per_query(self) -> int:
        return self.config.max_results_per_query

    @property
    def max_concurrent_searches(self) -> int:
        return self.config.max_concurrent_searches

    @property
    def enable_clarification(self) -> bool:
        return self.config.enable_clarification

    @property
    def enable_gap_review(self) -> bool:
        return self.config.enable_gap_review

    @property
    def enable_draft_review(self) -> bool:
        return self.config.enable_draft_review

    @property
    def max_findings_tokens(self) -> int:
        return self.config.max_findings_tokens
