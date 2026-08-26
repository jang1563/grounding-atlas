from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from single_cell_program_mask_transfer import (  # noqa: E402
    FORMS,
    _build_prompt,
    _equal_class_mean,
)


def test_equal_class_mean_does_not_weight_class_imbalance() -> None:
    labels = np.asarray([1, 1, 1, 0])
    values = np.asarray([0.2, 0.2, 0.2, 0.8])
    assert _equal_class_mean(values, labels) == 0.5


def test_prompt_factorial_crosses_order_and_queried_target() -> None:
    row = {
        "program_mask_sentence": "CD3D MASKED_GENE",
        "control_mask_sentence": "CD3D CONTROL",
    }
    prompts = {
        form: _build_prompt(row, "program", form)
        for form in FORMS
    }
    assert "Is it a CD8+ T cell or a NK cell?" in prompts["ab_pa"]
    assert "Is it a NK cell or a CD8+ T cell?" in prompts["ba_pa"]
    assert "probability that it is a CD8+ T cell" in prompts["ab_pa"]
    assert "probability that it is a NK cell" in prompts["ab_pb"]
    assert len(set(prompts.values())) == 4
