from pydantic import BaseModel


class PeakRecord(BaseModel):
    sample_id: str
    tension: float
    axis: str
    cable: int
    frequency: list[float]
    magnitude: list[float]
