import pytest

from eval.admet_contract import ENDPOINT_ORIENTATION, orient_labels


def test_all_scored_admet_endpoints_have_explicit_orientation():
    assert ENDPOINT_ORIENTATION == {
        "herg": "align",
        "cyp3a4": "align",
        "cyp2d6": "align",
        "ames": "oppose",
        "solubility": "oppose",
        "permeability": "oppose",
        "clearance": None,
    }


@pytest.mark.parametrize(
    ("endpoint", "labels", "expected"),
    [
        ("herg", [0, 1], [0, 1]),
        ("cyp3a4", [0, 1], [0, 1]),
        ("ames", [0, 1], [1, 0]),
        ("solubility", [0, 1], [1, 0]),
    ],
)
def test_orientation_maps_source_labels_to_yes_probability(endpoint, labels, expected):
    assert orient_labels(endpoint, labels).tolist() == expected


def test_unresolved_clearance_orientation_is_rejected():
    with pytest.raises(ValueError, match="resolved label orientation"):
        orient_labels("clearance", [0, 1])
