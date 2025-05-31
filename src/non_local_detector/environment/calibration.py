# calibration.py
import numpy as np

from .transforms import Affine2D


def simple_scale(
    px_per_cm: float, offset_px: tuple[float, float] = (0.0, 0.0)
) -> Affine2D:
    """
    Create a simple Affine2D transform that converts pixel units to centimeters.

    This returns an Affine2D matrix which, when applied to [x_px, y_px, 1]^T,
    yields coordinates in centimeters.

    Parameters
    ----------
    px_per_cm : float
        Number of pixels per centimeter. Must be nonzero.
    offset_px : tuple of two floats, optional (default: (0.0, 0.0))
        A pixel offset (x_offset, y_offset). The returned transform first
        subtracts this offset (in pixels) before scaling.

    Returns
    -------
    Affine2D
        An affine transformation representing:
            [ x_cm ]   [ 1/px_per_cm      0       -offset_px[0]/px_per_cm ] [ x_px ]
            [ y_cm ] = [      0       1/px_per_cm  -offset_px[1]/px_per_cm ] [ y_px ]
            [   1  ]   [      0            0            1                  ] [  1   ]

    Raises
    ------
    ValueError
        If `px_per_cm` is zero.
    """
    if px_per_cm == 0:
        raise ValueError("px_per_cm must be nonzero to avoid division by zero.")

    # Compute scale factors
    sx = sy = 1.0 / px_per_cm

    # Ensure offset_px has exactly two values
    try:
        ox, oy = float(offset_px[0]), float(offset_px[1])
    except (TypeError, IndexError):
        raise ValueError("offset_px must be a tuple of two numeric values (x, y).")

    # Build a 3×3 affine matrix: scale then translate
    tx = -ox * sx
    ty = -oy * sy
    A = np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return Affine2D(A)
