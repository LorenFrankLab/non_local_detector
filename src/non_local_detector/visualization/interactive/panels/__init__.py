"""Panel ABCs and (later) Qt panel implementations.

Phase 1c ships only the ABCs in ``base.py``. Qt panels live under
``panels/qt/`` and land starting in Phase 2.
"""

from non_local_detector.visualization.interactive.panels.base import (
    BinSyncedPanel,
    TimeAxisPanel,
)

__all__ = ["BinSyncedPanel", "TimeAxisPanel"]
