"""
non_local_detector.core.validation
----------------------------------

Centralized input checks; imported by core.EMDriver *and* exposed
to end-users through bundle.validate_sources().
"""

from typing import Iterable

from .bundle import DecoderBatch


def validate_sources(batch: DecoderBatch, models: Iterable) -> None:
    missing = set()
    for model in models:
        for src in getattr(model, "required_sources", ()):
            # direct attribute or signal dict
            if not hasattr(batch, src) and src not in batch.signals:
                missing.add(src)
    if missing:
        raise ValueError(
            "DecoderBatch is missing the following required fields:\n  • "
            + "\n  • ".join(sorted(missing))
        )
