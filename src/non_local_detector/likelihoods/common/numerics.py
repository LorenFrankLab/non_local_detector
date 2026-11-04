"""Numerical floors, the rate reference and the log of non-negative values."""

import jax
import jax.numpy as jnp
import numpy as np

# Internal module: no public API.
__all__: list[str] = []


EPS = 1e-15


LOG_EPS = np.log(EPS)


RATE_REFERENCE_SECONDS = 0.002


LOG_RATE_EPS_HZ = np.log(EPS / RATE_REFERENCE_SECONDS)


RATE_EPS_HZ = (
    EPS / RATE_REFERENCE_SECONDS
)  # The background rate beta of every active clusterless state, in Hz.


# Default clusterless mark bandwidth, in waveform-feature units.
DEFAULT_WAVEFORM_STD = 24.0


def log_nonnegative(x: jax.Array) -> jax.Array:
    """Log of a non-negative array: ``-inf`` at zero, ``NaN`` kept.

    No floor is applied, so a zero intensity stays an impossible event for the
    caller to combine (for example with a background by ``logaddexp``). The
    gradient is zero at zero instead of ``NaN``.

    Parameters
    ----------
    x : jax.Array
        Non-negative values.

    Returns
    -------
    log_x : jax.Array
        Same shape as ``x``.
    """
    positive = x > 0
    return jnp.where(
        positive,
        jnp.log(jnp.where(positive, x, 1.0)),
        jnp.where(jnp.isnan(x), jnp.nan, -jnp.inf),
    )
