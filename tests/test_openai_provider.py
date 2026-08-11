"""Unit tests for the generic OpenAI structured-output adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from presentation_pipeline.budgeting import serialize_provider_input
from presentation_pipeline.planning.models import (
    EvidenceSelection,
    OutlineSection,
    PresentationOutline,
    SelectedEvidence,
    SlideOutline,
    SlidePurpose,
)
from presentation_pipeline.providers import (
    OpenAIStructuredGenerationError,
    OpenAIStructuredGenerator,
)
from presentation_pipeline.understanding.models import DocumentDigest, EvidenceRef, KeyFact


class _FakeResponses:
    def __init__(self, response: object | BaseException) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class _FakeClient:
    def __init__(self, response: object | BaseException) -> None:
        self.responses = _FakeResponses(response)


def _response(*, parsed: object | None, status: str = "completed", **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        output_parsed=parsed,
        output=kwargs.pop("output", []),
        usage=kwargs.pop("usage", SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20)),
        _request_id=kwargs.pop("request_id", "req_123"),
        **kwargs,
    )


def _digest() -> DocumentDigest:
    return DocumentDigest(
        doc_id="doc-1",
        summary="summary",
        key_facts=[KeyFact(claim="fact", evidence=[EvidenceRef(doc_id="doc-1", evidence_ids=["evidence-1"])])],
    )


def _selection() -> EvidenceSelection:
    return EvidenceSelection(selected=[SelectedEvidence(doc_id="doc-1", evidence_id="evidence-1", reason="relevant")])


def _outline() -> PresentationOutline:
    return PresentationOutline(
        title="Title",
        objective="Objective",
        narrative="Narrative",
        sections=[
            OutlineSection(
                section_id="section-1",
                title="Section",
                purpose="Explain",
                slides=[
                    SlideOutline(
                        slide_id="slide-1",
                        title="Content",
                        purpose=SlidePurpose.CONTENT,
                        message="Message",
                        evidence=[EvidenceRef(doc_id="doc-1", evidence_ids=["evidence-1"])],
                    )
                ],
            )
        ],
    )


def _generate(generator: OpenAIStructuredGenerator, response_model: type[object]) -> object:
    return asyncio.run(
        generator.generate(
            system_prompt="Do the task.",
            input_data={"z": ["漢字", 2], "a": {"b": True}},
            response_model=response_model,  # type: ignore[arg-type]
        )
    )


@pytest.mark.parametrize("parsed", [_digest(), _selection(), _outline()])
def test_generates_each_pipeline_model_without_model_specific_branches(parsed: object) -> None:
    client = _FakeClient(_response(parsed=parsed))
    generator = OpenAIStructuredGenerator(model="gpt-test", client=client)

    result = _generate(generator, type(parsed))

    assert result is parsed
    assert client.responses.calls == [
        {
            "model": "gpt-test",
            "instructions": "Do the task.",
            "input": '{"a":{"b":true},"z":["漢字",2]}',
            "text_format": type(parsed),
            "store": False,
        }
    ]


def test_configures_store_and_preserves_deterministic_input_serialization() -> None:
    client = _FakeClient(_response(parsed=_digest()))
    generator = OpenAIStructuredGenerator(model="gpt-test", client=client, store=True)

    _generate(generator, DocumentDigest)

    call = client.responses.calls[0]
    assert call["store"] is True
    expected = json.dumps(
        {"a": {"b": True}, "z": ["漢字", 2]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert serialize_provider_input({"a": {"b": True}, "z": ["漢字", 2]}) == expected
    assert call["input"] == expected


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_rejects_any_non_completed_status(status: str) -> None:
    client = _FakeClient(_response(parsed=_digest(), status=status, incomplete_details=SimpleNamespace(reason="max_output_tokens")))
    generator = OpenAIStructuredGenerator(model="gpt-test", client=client)

    with pytest.raises(OpenAIStructuredGenerationError, match=f"status='{status}'.*max_output_tokens"):
        _generate(generator, DocumentDigest)


def test_rejects_refusal_and_completed_response_without_parsed_output() -> None:
    refused = _response(
        parsed=None,
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="policy")])],
    )
    generator = OpenAIStructuredGenerator(model="gpt-test", client=_FakeClient(refused))
    with pytest.raises(OpenAIStructuredGenerationError, match="refused: policy"):
        _generate(generator, DocumentDigest)

    no_parse = OpenAIStructuredGenerator(model="gpt-test", client=_FakeClient(_response(parsed=None)))
    with pytest.raises(OpenAIStructuredGenerationError, match="without parsed"):
        _generate(no_parse, DocumentDigest)


def test_sdk_exceptions_propagate_unchanged() -> None:
    error = ConnectionError("network unavailable")
    generator = OpenAIStructuredGenerator(model="gpt-test", client=_FakeClient(error))

    with pytest.raises(ConnectionError) as raised:
        _generate(generator, DocumentDigest)

    assert raised.value is error


def test_emits_content_free_telemetry_and_telemetry_failures_do_not_break_success() -> None:
    events: list[object] = []
    client = _FakeClient(_response(parsed=_digest()))
    generator = OpenAIStructuredGenerator(model="gpt-test", client=client, telemetry_handler=events.append)

    assert _generate(generator, DocumentDigest) == _digest()

    event = events[0]
    assert event.provider == "openai"
    assert event.model == "gpt-test"
    assert event.response_model == "DocumentDigest"
    assert event.request_id == "req_123"
    assert event.latency_ms >= 0
    assert (event.input_tokens, event.output_tokens, event.total_tokens) == (12, 8, 20)
    assert event.outcome == "success"
    assert event.error_type is None
    assert "Do the task" not in repr(event)
    assert "漢字" not in repr(event)

    broken_telemetry = OpenAIStructuredGenerator(
        model="gpt-test",
        client=_FakeClient(_response(parsed=_digest())),
        telemetry_handler=lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )
    assert _generate(broken_telemetry, DocumentDigest) == _digest()


def test_error_telemetry_is_content_free_and_preserves_original_sdk_error() -> None:
    events: list[object] = []
    error = RuntimeError("sdk error")
    generator = OpenAIStructuredGenerator(
        model="gpt-test",
        client=_FakeClient(error),
        telemetry_handler=events.append,
    )

    with pytest.raises(RuntimeError, match="sdk error"):
        _generate(generator, DocumentDigest)

    assert events[0].outcome == "error"
    assert events[0].error_type == "RuntimeError"
    assert events[0].request_id is None
