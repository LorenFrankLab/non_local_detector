"""Required-output keys for ``BackendAdapter.set_required_outputs``.

The viewer tells the backend which optional ``WindowPayload`` fields
the currently-visible panels consume so window-load workers can skip
fetching arrays no one will render. Keys correspond one-to-one with
the optional fields on ``WindowPayload`` (the required ``posterior``
is always loaded regardless).

These constants exist so panel-registration sites use a single
authoritative spelling instead of bare string literals. Adding a new
panel that needs a new output starts by adding an entry here — typos
fail at import time rather than silently in the data-loading hot
path.
"""

from __future__ import annotations

from enum import StrEnum


class RequiredOutput(StrEnum):
    """Keys ``BackendAdapter.set_required_outputs`` recognizes.

    Inheriting from :class:`enum.StrEnum` lets these values stand in
    for the bare strings the data source's ``if "likelihood" in
    required`` style guards check — set-membership works identically.
    """

    POSTERIOR = "posterior"
    LIKELIHOOD = "likelihood"
    PREDICTIVE = "predictive"
    STATE_PROBABILITIES = "state_probabilities"
    POSITION = "position"
    POSITION_2D = "position_2d"
