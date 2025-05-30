"""Class for representing an environment and a condition (trial type, etc.)"""

from dataclasses import dataclass
from typing import Union


@dataclass(order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to.

    Attributes
    ----------
    name : str, optional
    encoding_group : str or int, optional
    is_local : bool, optional
    is_no_spike : bool, optional

    """

    name: str = ""
    encoding_group: Union[str, int] = 0
    is_local: bool = False
    is_no_spike: bool = False

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.name, self.encoding_group) == (
            other.name,
            other.encoding_group,
        )
