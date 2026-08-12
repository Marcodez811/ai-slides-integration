"""Validated, immutable budgeting configuration and request guards."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from .counting import TokenCounter

if TYPE_CHECKING:
    from presentation_pipeline.generation import StructuredGenerator


def _integer(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if not positive and value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value


def _input_budget(name: str, value: object) -> InputBudget:
    if not isinstance(value, InputBudget):
        raise TypeError(f"{name} must be an InputBudget")
    return value


@dataclass(frozen=True, slots=True)
class InputBudget:
    max_input_tokens: int
    output_reserve_tokens: int = 0
    safety_margin_tokens: int = 0
    target_input_tokens: int | None = None

    def __post_init__(self) -> None:
        _integer("max_input_tokens", self.max_input_tokens, positive=True)
        _integer("output_reserve_tokens", self.output_reserve_tokens)
        _integer("safety_margin_tokens", self.safety_margin_tokens)
        if self.usable_input_tokens <= 0:
            raise ValueError("output reserve and safety margin must leave usable input tokens")
        if self.target_input_tokens is not None:
            _integer("target_input_tokens", self.target_input_tokens, positive=True)
            if self.target_input_tokens > self.usable_input_tokens:
                raise ValueError("target_input_tokens must not exceed usable_input_tokens")

    @property
    def usable_input_tokens(self) -> int:
        return self.max_input_tokens - self.output_reserve_tokens - self.safety_margin_tokens

    @property
    def target_input_tokens_or_usable(self) -> int:
        return (
            self.target_input_tokens
            if self.target_input_tokens is not None
            else self.usable_input_tokens
        )

    @property
    def usable_tokens(self) -> int:
        return self.usable_input_tokens

    @property
    def target_tokens(self) -> int:
        return self.target_input_tokens_or_usable


def _default_window() -> InputBudget:
    return InputBudget(max_input_tokens=32_000, safety_margin_tokens=2_000)


def _default_stage() -> InputBudget:
    return InputBudget(max_input_tokens=32_000, safety_margin_tokens=2_000)


@dataclass(frozen=True, slots=True)
class UnderstandingBudgets:
    window: InputBudget = _default_window()
    reduction: InputBudget = _default_stage()

    def __post_init__(self) -> None:
        _input_budget("window", self.window)
        _input_budget("reduction", self.reduction)

    @property
    def document_window(self) -> InputBudget:
        return self.window

    @property
    def document_reduction(self) -> InputBudget:
        return self.reduction


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    request_budget: InputBudget = _default_stage()
    reduction_budget: InputBudget = _default_stage()
    selection_budget: InputBudget = _default_stage()
    max_candidates_per_window: int = 24
    max_global_candidates: int = 96

    def __post_init__(self) -> None:
        _input_budget("request_budget", self.request_budget)
        _input_budget("reduction_budget", self.reduction_budget)
        _input_budget("selection_budget", self.selection_budget)
        _integer(
            "max_candidates_per_window",
            self.max_candidates_per_window,
            positive=True,
        )
        _integer("max_global_candidates", self.max_global_candidates, positive=True)

    @property
    def candidate_discovery_budget(self) -> InputBudget:
        return self.request_budget


@dataclass(frozen=True, slots=True)
class PlanningBudgets:
    """Conservative application-level limits; not provider context limits."""

    understanding: UnderstandingBudgets = UnderstandingBudgets()
    retrieval: RetrievalBudget = RetrievalBudget()
    outline_generation: InputBudget = _default_stage()

    def __post_init__(self) -> None:
        if not isinstance(self.understanding, UnderstandingBudgets):
            raise TypeError("understanding must be an UnderstandingBudgets")
        if not isinstance(self.retrieval, RetrievalBudget):
            raise TypeError("retrieval must be a RetrievalBudget")
        _input_budget("outline_generation", self.outline_generation)


class InputBudgetExceededError(ValueError):
    """Raised before a provider request that would exceed its input allocation."""

    def __init__(self, *, stage: str, estimated_tokens: int, limit_tokens: int) -> None:
        self.stage = stage
        self.estimated_tokens = estimated_tokens
        self.limit_tokens = limit_tokens
        super().__init__(
            "input budget exceeded "
            f"(stage={stage!r}, estimated_tokens={estimated_tokens}, limit_tokens={limit_tokens})"
        )


def estimate_request_tokens(
    system_prompt: str,
    input_data: dict[str, object],
    token_counter: TokenCounter,
) -> int:
    """Estimate both trusted instructions and canonical serialized input data."""
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string")
    # ``count_payload`` is required to use the package's canonical serializer.
    return token_counter.count_text(system_prompt) + token_counter.count_payload(input_data)


def enforce_input_budget(
    *,
    input_data: dict[str, object],
    token_counter: TokenCounter,
    budget: InputBudget,
    stage: str,
    system_prompt: str = "",
) -> int:
    """Return a request estimate or raise a content-free over-budget error."""
    if not isinstance(budget, InputBudget):
        raise TypeError("budget must be an InputBudget")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-blank string")
    estimated_tokens = estimate_request_tokens(system_prompt, input_data, token_counter)
    if estimated_tokens > budget.usable_input_tokens:
        raise InputBudgetExceededError(
            stage=stage,
            estimated_tokens=estimated_tokens,
            limit_tokens=budget.usable_input_tokens,
        )
    return estimated_tokens


TStructured = TypeVar("TStructured", bound=BaseModel)


class GenerationLimiter:
    """One shared semaphore and hard preflight guard for provider generation."""

    def __init__(
        self,
        generator: "StructuredGenerator",
        *,
        token_counter: TokenCounter,
        concurrency: int = 4,
    ) -> None:
        _integer("concurrency", concurrency, positive=True)
        self._generator = generator
        self._token_counter = token_counter
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def token_counter(self) -> TokenCounter:
        """The counter shared by all limiter consumers."""
        return self._token_counter

    async def invoke(
        self,
        *,
        system_prompt: str,
        input_data: dict[str, object],
        response_model: type[TStructured],
        budget: InputBudget,
        stage: str,
    ) -> TStructured:
        from presentation_pipeline.generation import invoke_structured

        estimated_tokens = enforce_input_budget(
            input_data=input_data,
            system_prompt=system_prompt,
            token_counter=self._token_counter,
            budget=budget,
            stage=stage,
        )
        from presentation_pipeline.observability import (
            bind_limiter_call,
            record_limiter_call,
            reset_limiter_call,
            safe_error,
            safe_event,
        )

        record_limiter_call(stage=stage)
        safe_event(
            "llm_call_start",
            stage=stage,
            response_model=response_model.__name__,
            estimated_input_tokens=estimated_tokens,
            input_limit_tokens=budget.usable_input_tokens,
        )
        started = time.perf_counter()
        correlation = bind_limiter_call(
            stage=stage,
            response_model=response_model.__name__,
            estimated_input_tokens=estimated_tokens,
        )
        try:
            async with self._semaphore:
                result = await invoke_structured(
                    self._generator, system_prompt, input_data, response_model
                )
        except BaseException as error:
            # Carry only the pipeline stage forward for failure reports. This
            # avoids wrapping provider/domain exceptions or exposing payloads.
            if not hasattr(error, "stage"):
                try:
                    setattr(error, "stage", stage)
                except (AttributeError, TypeError):
                    pass
            safe_error(
                "llm_call_error",
                stage=stage,
                response_model=response_model.__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                outcome="error",
                error_type=type(error).__name__,
            )
            raise
        finally:
            reset_limiter_call(correlation)
        safe_event(
            "llm_call_end",
            stage=stage,
            response_model=response_model.__name__,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            outcome="success",
        )
        return result
