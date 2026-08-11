"""OpenAI Responses API adapter for provider-neutral structured generation."""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel


TStructured = TypeVar("TStructured", bound=BaseModel)
TelemetryHandler = Callable[["OpenAIGenerationTelemetry"], object | Awaitable[object]]


class OpenAIStructuredGenerationError(RuntimeError):
    """An API response that could not produce usable structured output."""


@dataclass(frozen=True, slots=True)
class OpenAIGenerationTelemetry:
    """Content-free metadata about one structured-generation attempt."""

    provider: str
    model: str
    response_model: str
    request_id: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    outcome: str
    error_type: str | None = None


class OpenAIStructuredGenerator:
    """Generate a Pydantic response with OpenAI Structured Outputs.

    The adapter deliberately knows nothing about pipeline response models.  The
    supplied Pydantic class is passed directly to the OpenAI SDK parser.
    """

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        store: bool = False,
        telemetry_handler: TelemetryHandler | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        self.model = model
        self.client = client if client is not None else AsyncOpenAI()
        self.store = store
        self.telemetry_handler = telemetry_handler

    async def generate(
        self,
        *,
        system_prompt: str,
        input_data: dict[str, object],
        response_model: type[TStructured],
    ) -> TStructured:
        """Call Responses ``parse`` and return its schema-validated result.

        SDK transport/authentication exceptions intentionally propagate unchanged.
        Provider-state failures, refusals, and missing parsed output become a
        small controlled error type that callers can handle consistently.
        """
        serialized_input = json.dumps(
            input_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        started = time.perf_counter()
        response: Any | None = None
        try:
            response = await self.client.responses.parse(
                model=self.model,
                instructions=system_prompt,
                input=serialized_input,
                text_format=response_model,
                store=self.store,
            )
            status = getattr(response, "status", None)
            if status != "completed":
                details = _response_failure_details(response)
                raise OpenAIStructuredGenerationError(
                    f"OpenAI response did not complete (status={status!r}{details})"
                )
            refusal = _find_refusal(response)
            if refusal is not None:
                raise OpenAIStructuredGenerationError(f"OpenAI response was refused: {refusal}")
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise OpenAIStructuredGenerationError(
                    "OpenAI response completed without parsed structured output"
                )
        except Exception as error:
            await self._emit_telemetry(
                response=response,
                response_model=response_model,
                started=started,
                outcome="error",
                error_type=type(error).__name__,
            )
            raise

        await self._emit_telemetry(
            response=response,
            response_model=response_model,
            started=started,
            outcome="success",
        )
        return parsed

    async def _emit_telemetry(
        self,
        *,
        response: Any | None,
        response_model: type[BaseModel],
        started: float,
        outcome: str,
        error_type: str | None = None,
    ) -> None:
        """Best-effort telemetry that never alters generation behavior."""
        if self.telemetry_handler is None:
            return
        usage = getattr(response, "usage", None)
        event = OpenAIGenerationTelemetry(
            provider="openai",
            model=self.model,
            response_model=response_model.__name__,
            request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            outcome=outcome,
            error_type=error_type,
        )
        try:
            result = self.telemetry_handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Observability is never allowed to turn a successful request into a
            # failure, nor to hide the original provider exception on failure.
            pass


def _response_failure_details(response: Any) -> str:
    details = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
    if details is None:
        return ""
    reason = getattr(details, "reason", None) or getattr(details, "code", None)
    return f", reason={reason!r}" if reason else ""


def _find_refusal(response: Any) -> str | None:
    """Extract a refusal reason across the SDK's response-content shapes."""
    direct_refusal = getattr(response, "refusal", None)
    if direct_refusal:
        return str(direct_refusal)
    for output in getattr(response, "output", ()) or ():
        output_refusal = getattr(output, "refusal", None)
        if output_refusal:
            return str(output_refusal)
        for content in getattr(output, "content", ()) or ():
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
            if getattr(content, "type", None) == "refusal":
                return str(getattr(content, "text", None) or "policy refusal")
    return None
