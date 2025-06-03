"""
transforms.py - minimal 2-D coordinate transforms
=================================================

Two complementary APIs
----------------------
1.  *Composable objects* (`Affine2D`, `SpatialTransform`)
    Build a transform once, reuse everywhere, keep provenance.
2.  *Quick helpers* (`flip_y_data`, `convert_to_cm`, `convert_to_pixels`)
    One-liners for scripts that just need a NumPy array back.

All functions assume coordinates are shaped ``(..., 2)`` and are no-ops on
the x-axis unless you chain additional transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------
# 1.  Composable transform objects
# ---------------------------------------------------------------------
@runtime_checkable
class SpatialTransform(Protocol):
    """Callable that maps an (N, 2) array of points → (N, 2) array."""

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class Affine2D(SpatialTransform):
    """
    2-D affine transform expressed as a 3 × 3 homogeneous matrix *A* such that

        [x', y', 1]^T  =  A @ [x, y, 1]^T
    """

    A: NDArray[np.float64]  # shape (3, 3)

    # ---- core --------------------------------------------------------
    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        pts = np.asanyarray(pts, dtype=float)
        pts_h = np.c_[pts.reshape(-1, 2), np.ones((pts.size // 2, 1))]
        out = pts_h @ self.A.T
        out = out[:, :2] / out[:, 2:3]
        return out.reshape(pts.shape)

    # ---- helpers -----------------------------------------------------
    def inverse(self) -> "Affine2D":
        """Return the inverse transform."""
        return Affine2D(np.linalg.inv(self.A))

    def compose(self, other: "Affine2D") -> "Affine2D":
        """Return ``self ∘ other`` (apply *other* first, then *self*)."""
        return Affine2D(self.A @ other.A)

    # Pythonic shorthand:  t3 = t1 @ t2
    def __matmul__(self, other: "Affine2D") -> "Affine2D":  # noqa: D401
        return self.compose(other)


def identity() -> Affine2D:
    """Return the identity transform."""
    return Affine2D(np.eye(3))


# Factory helpers for the most common ops ---------------------------------
def scale_2d(sx: float = 1.0, sy: Optional[float] = None) -> Affine2D:
    """Uniform or anisotropic scaling."""
    sy = sx if sy is None else sy
    return Affine2D(np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]))


def translate(tx: float = 0.0, ty: float = 0.0) -> Affine2D:
    """Translation by (*tx*, *ty*)."""
    return Affine2D(np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]))


def flip_y(frame_height_px: float) -> Affine2D:
    """
    Flip the *y*-axis of pixel coordinates so that origin moves
    from top-left to bottom-left.

    Parameters
    ----------
    frame_height_px
        Height of the video frame in **pixels**.
    """
    return Affine2D(
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, frame_height_px], [0.0, 0.0, 1.0]])
    )


# ---------------------------------------------------------------------
# Quick NumPy helpers that *internally* build and apply Affine2D
# ---------------------------------------------------------------------
def flip_y_data(
    data: Union[NDArray[np.float64], tuple, list],
    frame_size_px: tuple[float, float],
) -> NDArray[np.float64]:
    """
    Flip y-axis of coordinates so that the origin moves from
    image-space top-left to Cartesian bottom-left.

    Equivalent to::

        Affine2D([[1, 0, 0],
                  [0,-1,H],
                  [0, 0,1]])(data)

    but without the user having to build the transform.
    """
    transform = flip_y(frame_height_px=frame_size_px[1])
    return transform(np.asanyarray(data, dtype=float))


def convert_to_cm(
    data_px: Union[NDArray[np.float64], tuple, list],
    frame_size_px: tuple[float, float],
    cm_per_px: float = 1.0,
) -> NDArray[np.float64]:
    """Convert pixel coordinates to centimeter coordinates.

    Pixel  →  centimeter coordinates *and* y-flip in one shot.

    Internally constructs ``scale_2d(cm_per_px) @ flip_y(H)`` and applies it.

    Parameters
    ----------
    data_px : array-like
        Input coordinates in pixel space, shape (..., 2).
    frame_size_px : tuple[float, float]
        Size of the video frame in pixels (width, height).
    cm_per_px : float, optional
        Conversion factor from pixels to centimeters (default is 1.0).

    Returns
    -------
    NDArray[np.float64]
        Converted coordinates in centimeters, shape (..., 2).

    """
    T = scale_2d(cm_per_px) @ flip_y(frame_height_px=frame_size_px[1])
    return T(np.asanyarray(data_px, dtype=float))


def convert_to_pixels(
    data_cm: Union[NDArray[np.float64], tuple, list],
    frame_size_px: tuple[float, float],
    cm_per_px: float = 1.0,
) -> NDArray[np.float64]:
    """
    Centimeter  →  pixel coordinates with y-flip (inverse of `convert_to_cm`).

    Internally constructs ``flip_y(H) @ scale_2d(1/cm_per_px)``.
    """
    T = flip_y(frame_height_px=frame_size_px[1]) @ scale_2d(1.0 / cm_per_px)
    return T(np.asanyarray(data_cm, dtype=float))
