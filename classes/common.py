from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Cables(Enum):
    C1 = auto()
    C2 = auto()
    C3 = auto()
    C4 = auto()


class Axes(Enum):
    X = "x"
    Y = "y"
    Z = "z"


@dataclass
class Id:
    cable: Optional[Cables] = None
    axis: Optional[Axes] = None
