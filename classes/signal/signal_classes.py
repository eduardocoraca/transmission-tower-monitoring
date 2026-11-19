from typing import Optional, List
from pydantic import BaseModel


class SignalFRF(BaseModel):
    frequencies: List[float]
    abs: List[float]
    angle: List[float]


class SignalSpectrum(BaseModel):
    frequencies: List[float]
    magnitudes: List[float]


class WaveletPacket(BaseModel):
    max_level: int
    family: str
    fs: float
    energy_map: List[float]
    entropy_map: List[float]
