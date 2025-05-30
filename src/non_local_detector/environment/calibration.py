# calibration.py
import numpy as np

from .transforms import Affine2D


def simple_scale(px_per_cm: float, offset_px: tuple[float, float] = (0, 0)) -> Affine2D:
    sx = sy = 1.0 / px_per_cm
    tx, ty = -np.asarray(offset_px) * sx
    A = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])
    return Affine2D(A)
