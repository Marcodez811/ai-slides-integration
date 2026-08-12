"""Finite provider budgets for semantic slide-content generation."""

from __future__ import annotations

from dataclasses import dataclass

from presentation_pipeline.budgeting import InputBudget


def _slide_content_budget() -> InputBudget:
    return InputBudget(max_input_tokens=24_000, safety_margin_tokens=2_000)


@dataclass(frozen=True, slots=True)
class SynthesisBudgets:
    slide_content: InputBudget = _slide_content_budget()

    def __post_init__(self) -> None:
        if not isinstance(self.slide_content, InputBudget):
            raise TypeError("slide_content must be an InputBudget")
