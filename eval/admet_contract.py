"""Shared ADMET label-to-question orientation contract.

The committed NegResultDB-derived rows use source label 1 for an aggregated assay ``fail``.
That source label is not always the semantic "yes" answer to the question shown to a model.
Keep the transformation in one place so transfer arms cannot silently pool incompatible targets.
"""

import numpy as np

ENDPOINT_ORIENTATION = {
    "herg": "align",
    "cyp3a4": "align",
    "cyp2d6": "align",
    "ames": "oppose",
    "solubility": "oppose",
    "permeability": "oppose",
    # The source combines heterogeneous units and the direction has not been resolved.
    "clearance": None,
}


def orient_label(endpoint, source_label):
    """Return 1 exactly when the endpoint-specific natural-language question is true."""
    orientation = ENDPOINT_ORIENTATION.get(endpoint)
    if orientation is None:
        raise ValueError(f"endpoint {endpoint!r} has no resolved label orientation")
    label = int(source_label)
    if label not in (0, 1):
        raise ValueError(f"source label must be binary, got {source_label!r}")
    return 1 - label if orientation == "oppose" else label


def orient_labels(endpoint, source_labels):
    """Vector form of :func:`orient_label`."""
    labels = np.asarray(source_labels, dtype=int)
    return np.asarray([orient_label(endpoint, label) for label in labels], dtype=int)
