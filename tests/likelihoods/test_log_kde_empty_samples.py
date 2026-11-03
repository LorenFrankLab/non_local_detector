"""An empty KDE floors to ``LOG_EPS`` instead of NaN.

A zero-encoding-spike electrode (a unit that never fired during training) has an
empty mark-density KDE. Evaluating ``log(density)`` over an empty sample set as
``logsumexp(empty) - logsumexp(empty)`` is ``-inf - -inf = NaN``, which the HMM
filter would propagate into the posterior. That electrode's decode-spike terms
are checked against the background for every algorithm in
``test_clusterless_mark_background.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from non_local_detector.likelihoods.common import (
    LOG_EPS,
    block_log_kde,
    log_kde,
)


@pytest.mark.unit
def test_log_kde_empty_samples_floors_to_log_eps():
    """The empty-KDE NaN source is guarded at the ``log_kde`` level.

    With no samples, ``log_den = logsumexp(empty) = -inf`` and a naive
    ``log_num - log_den`` is ``-inf - -inf = NaN``. ``log_kde`` instead returns
    the finite ``LOG_EPS`` floor (via its ``isneginf(log_den)`` guard), so a
    zero-encoding-spike electrode contributes a floor, not a NaN.
    """
    eval_points = jnp.asarray(
        np.random.default_rng(0).standard_normal((5, 3)), dtype=jnp.float32
    )
    empty_samples = jnp.zeros((0, 3))
    std = jnp.ones(3)

    direct = np.asarray(log_kde(eval_points, empty_samples, std, jnp.zeros((0,))))
    blocked = np.asarray(
        block_log_kde(eval_points=eval_points, samples=empty_samples, std=std)
    )
    for out in (direct, blocked):
        assert np.all(np.isfinite(out))
        assert_allclose(out, LOG_EPS)
