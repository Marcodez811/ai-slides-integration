"""Provider-neutral structured generation boundary."""

from __future__ import annotations

import inspect
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


TStructured = TypeVar("TStructured", bound=BaseModel)


@runtime_checkable
class StructuredGenerator(Protocol):
    """Minimal adapter boundary for a structured-output LLM provider."""

    async def generate(
        self,
        *,
        system_prompt: str,
        input_data: dict[str, object],
        response_model: type[TStructured],
    ) -> TStructured | dict[str, object]: ...


async def invoke_structured(
    generator: StructuredGenerator,
    system_prompt: str,
    input_data: dict[str, object],
    response_model: type[TStructured],
) -> TStructured:
    """Call a provider adapter and normalize its response with Pydantic."""
    result = generator.generate(
        system_prompt=system_prompt,
        input_data=input_data,
        response_model=response_model,
    )
    if inspect.isawaitable(result):
        result = await result
    return response_model.model_validate(result)
