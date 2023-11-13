from abc import ABC, abstractmethod
from typing import Any


class Signal(ABC):
    @abstractmethod
    def get(self) -> Any:
        pass
