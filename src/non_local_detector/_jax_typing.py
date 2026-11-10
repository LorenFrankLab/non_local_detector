"""Typed wrappers of the JAX functions that ship without annotations.

mypy treats a call to an unannotated function as untyped
(``no-untyped-call``). Wrapping each one here keeps that suppression in one
place and gives callers a precise return type.
"""

from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

_T = TypeVar("_T")


def finfo(dtype: DTypeLike) -> np.finfo[Any]:
    """Machine limits of a floating-point dtype, as ``jax.numpy.finfo``.

    Parameters
    ----------
    dtype : dtype
        A floating-point dtype, including the JAX-only ones (``bfloat16``).

    Returns
    -------
    finfo : np.finfo
        Its limits (``eps``, ``tiny``, ``max``, ...).
    """
    return jnp.finfo(dtype)  # type: ignore[no-untyped-call]  # untyped in JAX


def optimization_barrier(operand: _T) -> _T:
    """``jax.lax.optimization_barrier``: stop XLA reordering across ``operand``.

    Parameters
    ----------
    operand : pytree of arrays
        The values XLA must compute before anything that uses them.

    Returns
    -------
    operand : pytree of arrays
        The same values.
    """
    return jax.lax.optimization_barrier(operand)  # type: ignore[no-any-return,no-untyped-call]  # untyped in JAX
