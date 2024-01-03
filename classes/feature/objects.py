from dataclasses import dataclass

import numpy as np


@dataclass
class Pole:
    z: complex
    a: complex
    fs: float

    @property
    def abs(self):
        return np.abs(self.z)

    @property
    def wn(self):
        p = np.angle(self.z)
        return np.abs(p)

    @property
    def fn(self):
        return self.fs * self.wn / (2 * np.pi)


@dataclass
class Poles:
    p: list[Pole]
