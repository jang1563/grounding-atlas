import pytest

from eval import budget_arm
from eval.budget_arm import parse_final_probability


@pytest.mark.parametrize(
    ("text", "probability", "valid"),
    [
        ("Reasoning with numbers 2 and 4.\nProbability: 0.73", 0.73, True),
        ("Probability: .4", 0.4, True),
        ("Probability: 73", 0.5, False),
        ("Probability: 0.7\nextra text", 0.5, False),
        ("Reasoning only: 0.7", 0.5, False),
    ],
)
def test_budget_parser_requires_the_declared_final_line(text, probability, valid):
    parsed, ok = parse_final_probability(text)
    assert parsed == pytest.approx(probability)
    assert ok is valid


def test_budget_completion_uses_reasoning_system_contract(monkeypatch):
    observed = {}

    def fake_complete(model, prompt, image=None, system=None):
        observed.update(model=model, prompt=prompt, image=image, system=system)
        return "Probability: 0.6"

    monkeypatch.setattr(budget_arm.rge, "complete", fake_complete)

    output = budget_arm.complete_reasoning("model", "prompt", image="image.png")

    assert output == "Probability: 0.6"
    assert observed["system"] == budget_arm.REASON_SYSTEM
