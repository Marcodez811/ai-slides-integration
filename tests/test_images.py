"""Offline contracts for optional decorative image generation."""

from presentation_pipeline.images import illustration_prompt


def test_illustration_prompt_forbids_evidentiary_visuals() -> None:
    prompt = illustration_prompt(
        title="Health policy",
        message="Improve access to care",
        audience="Leadership",
        tone=None,
    )
    assert "no words" in prompt
    assert "numbers" in prompt
    assert "data charts" in prompt
    assert "Health policy" in prompt
    assert "Leadership" in prompt
    assert "restrained executive-policy briefing" in prompt
