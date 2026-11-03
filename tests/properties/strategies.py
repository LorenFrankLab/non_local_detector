"""Hypothesis strategies shared by the property-based tests."""

import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis import strategies as st


@st.composite
def probability_distribution(draw, min_size=2, max_size=20):
    """Generate valid probability distributions.

    Returns an array of positive values that sum to 1.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate positive values
    values = draw(
        npst.arrays(
            dtype=np.float64,
            shape=(size,),
            elements=st.floats(
                min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Normalize to sum to 1
    return values / values.sum()


@st.composite
def stochastic_matrix(draw, min_size=2, max_size=10):
    """Generate valid stochastic (row-normalized) matrices.

    Each row sums to 1, representing valid transition probabilities.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    matrix = draw(
        npst.arrays(
            dtype=np.float64,
            shape=(size, size),
            elements=st.floats(
                min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Row-normalize
    return matrix / matrix.sum(axis=1, keepdims=True)
