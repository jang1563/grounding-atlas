from __future__ import annotations

import math

from eval import analyze_coherent_readout_v5_fit_posthoc as posthoc


def _row(*, world: str, p: int, v: int, margin: float, correct: bool) -> dict:
    return {
        "world_id": world,
        "factors": {"p": p, "v": v},
        "diagnostics": {
            "answer_correct": correct,
            "expected_minus_distractor_margin": margin,
        },
    }


def test_factorial_term_summary_preserves_world_signs() -> None:
    rows = [
        _row(world="w1", p=-1, v=-1, margin=-2.0, correct=False),
        _row(world="w1", p=-1, v=1, margin=2.0, correct=True),
        _row(world="w1", p=1, v=-1, margin=2.0, correct=True),
        _row(world="w1", p=1, v=1, margin=-2.0, correct=False),
        _row(world="w2", p=-1, v=-1, margin=-1.0, correct=False),
        _row(world="w2", p=-1, v=1, margin=1.0, correct=True),
        _row(world="w2", p=1, v=-1, margin=1.0, correct=True),
        _row(world="w2", p=1, v=1, margin=-1.0, correct=False),
    ]

    result = posthoc.factorial_term_summary(rows, ("p", "v"))

    assert result["margin_beta"] == -1.5
    assert result["negative_worlds"] == 2
    assert result["positive_worlds"] == 0
    assert result["world_margin_betas"] == {"w1": -2.0, "w2": -1.0}


def test_exact_mcnemar_two_sided_handles_one_way_discordance() -> None:
    assert posthoc.exact_mcnemar_two_sided(0, 33) == 2.0**-32
    assert posthoc.exact_mcnemar_two_sided(0, 0) == 1.0
    assert math.isclose(posthoc.exact_mcnemar_two_sided(2, 8), 0.109375)
