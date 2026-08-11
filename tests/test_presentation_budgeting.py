"""Offline tests for provider-neutral request budgeting."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from presentation_pipeline.budgeting import (
    CharacterTokenEstimator,
    GenerationLimiter,
    InputBudget,
    InputBudgetExceededError,
    PlanningBudgets,
    RetrievalBudget,
    UnderstandingBudgets,
    Utf8ByteTokenEstimator,
    enforce_input_budget,
    estimate_request_tokens,
    serialize_provider_input,
)


def test_serialization_is_deterministic_and_shared() -> None:
    assert serialize_provider_input({"z": ["漢字", 2], "a": {"b": True}}) == '{"a":{"b":true},"z":["漢字",2]}'


def test_character_estimator_counts_text_and_canonical_payload() -> None:
    counter = CharacterTokenEstimator(chars_per_token=4)
    assert counter.count_text("") == 0
    assert counter.count_text("abcde") == 2
    assert counter.count_payload({"z": 1, "a": True}) == 4


def test_utf8_byte_estimator_is_conservative_for_multibyte_text() -> None:
    counter = Utf8ByteTokenEstimator()
    assert counter.count_text("漢字") == 6
    assert counter.count_payload({"value": "漢"}) >= len('{"value":"漢"}'.encode("utf-8"))


@pytest.mark.parametrize("kwargs", [
    {"max_input_tokens": 0},
    {"max_input_tokens": -1},
    {"max_input_tokens": True},
    {"max_input_tokens": 10, "output_reserve_tokens": True},
    {"max_input_tokens": 10, "safety_margin_tokens": -1},
    {"max_input_tokens": 10, "output_reserve_tokens": 7, "safety_margin_tokens": 3},
    {"max_input_tokens": 10, "target_input_tokens": 0},
    {"max_input_tokens": 10, "target_input_tokens": True},
    {"max_input_tokens": 10, "safety_margin_tokens": 2, "target_input_tokens": 9},
])
def test_input_budget_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        InputBudget(**kwargs)  # type: ignore[arg-type]


def test_input_budget_exposes_usable_and_target_properties() -> None:
    budget = InputBudget(100, output_reserve_tokens=20, safety_margin_tokens=5, target_input_tokens=50)
    assert budget.usable_input_tokens == budget.usable_tokens == 75
    assert budget.target_input_tokens_or_usable == budget.target_tokens == 50
    assert InputBudget(10).target_input_tokens_or_usable == 10


def test_stage_budget_defaults_are_finite_and_candidate_caps_are_validated() -> None:
    assert PlanningBudgets().understanding.window.usable_input_tokens > 0
    assert PlanningBudgets().retrieval.selection_budget.usable_input_tokens > 0
    assert PlanningBudgets().outline_generation.usable_input_tokens > 0
    assert UnderstandingBudgets().reduction.usable_input_tokens > 0
    assert RetrievalBudget().max_global_candidates > 0
    with pytest.raises(TypeError):
        RetrievalBudget(max_candidates_per_window=True)


def test_request_estimation_counts_prompt_and_payload() -> None:
    class Counter:
        def count_text(self, text: str) -> int:
            return len(text)

        def count_payload(self, payload: dict[str, object]) -> int:
            assert payload == {"value": 2}
            return 7

    assert estimate_request_tokens("abcd", {"value": 2}, Counter()) == 11


def test_budget_guard_rejects_oversized_payload_without_echoing_content() -> None:
    secret = "do-not-echo-this-document-content"
    with pytest.raises(InputBudgetExceededError) as raised:
        enforce_input_budget(
            system_prompt="prompt",
            input_data={"secret": secret},
            token_counter=CharacterTokenEstimator(chars_per_token=1),
            budget=InputBudget(3),
            stage="digest",
        )
    error = raised.value
    assert (error.stage, error.limit_tokens) == ("digest", 3)
    assert secret not in str(error)
    assert "estimated_tokens" in str(error)


class _Output(BaseModel):
    value: str


class _SlowGenerator:
    def __init__(self) -> None:
        self.active = 0
        self.maximum = 0

    async def generate(self, **_: object) -> _Output:
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return _Output(value="ok")


def test_generation_limiter_enforces_budget_before_provider_and_shared_concurrency() -> None:
    async def run() -> None:
        generator = _SlowGenerator()
        limiter = GenerationLimiter(generator, token_counter=CharacterTokenEstimator(), concurrency=1)
        await asyncio.gather(*(
            limiter.invoke(
                system_prompt="p",
                input_data={"item": item},
                response_model=_Output,
                budget=InputBudget(100),
                stage="test",
            )
            for item in range(3)
        ))
        assert generator.maximum == 1
        with pytest.raises(InputBudgetExceededError):
            await limiter.invoke(
                system_prompt="p",
                input_data={"secret": "x" * 200},
                response_model=_Output,
                budget=InputBudget(1),
                stage="test",
            )
        assert generator.active == 0

    asyncio.run(run())
