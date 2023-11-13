from dataclasses import dataclass


@dataclass
class Poles:
    z: complex
    f: float
    w: list[float]
    r: list[float]
