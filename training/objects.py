from typing import Literal, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# -- Feature-based objects --


class Embedding(BaseModel):
    cable: int
    sampled_at: datetime
    tension: float
    values: list[float]


class SpectralFeatures(BaseModel):
    sampled_at: datetime
    id: str
    tension: float
    cable: Literal["c1", "c2", "c3", "c4"]
    level: int
    energy: float
    norm_energy: float | None = None


class NewSpectralFeatures(BaseModel):
    sampled_at: datetime
    tension: float
    cable: Literal["c1", "c2", "c3", "c4"]
    values: list[float]
    label: Optional[int] = None


class BandRange(BaseModel):
    start: float
    end: float

    @property
    def bandwidth(self) -> float:
        return self.end - self.start

    @property
    def id(self) -> str:
        return f"{self.start}-{self.end}"

    def has_intersection(self, other: "BandRange") -> bool:
        """Check if this frequency band range intersects with another range.

        Args:
            other (BandRange): The other frequency band range to check intersection with

        Returns:
            bool: True if the ranges intersect, False otherwise

        Examples:
            >>> r1 = BandRange(start=0, end=10)
            >>> r2 = BandRange(start=5, end=15)
            >>> r1.has_intersection(r2)
            True
            >>> r3 = BandRange(start=20, end=30)
            >>> r1.has_intersection(r3)
            False
        """
        if other.start >= self.end:
            return False

        if self.start >= other.end:
            return False

        if other.start > self.start and self.end <= other.end:
            return True

        if self.start > other.start and other.end <= self.end:
            return True

        if other.start > self.start and other.end <= self.end:
            return True

        if self.start > other.start and self.end <= other.end:
            return True
        return False


class WptCableDataset(BaseModel):
    wpt: np.ndarray
    tension: np.ndarray
    label: np.ndarray
    sampled_at: list[datetime]
    cable: str

    class Config:
        arbitrary_types_allowed = True


class ScalerParams(BaseModel):
    mu: list[float]
    sigma: list[float]


class PipelineOutput(BaseModel):
    train_vae_results: list[Embedding]
    valid_vae_results: list[Embedding]
    test_vae_results: Optional[list[Embedding]] = None
    train_dist_results: Optional[list[Embedding]] = None
    valid_dist_results: Optional[list[Embedding]] = None
    test_dist_results: Optional[list[Embedding]] = None
    model: Optional[Any] = None
    train_mu: Optional[list[float]] = None
    train_std: Optional[list[float]] = None


@dataclass
class VAEResults:
    x: np.ndarray
    z: np.ndarray
    mse: np.ndarray
    rec: np.ndarray
    tensions: np.ndarray
    dates: list[datetime]


# -- Cosine-based objects --


class MetaData(BaseModel):
    cable: str
    tension: float
    sampled_at: datetime


@dataclass
class CosinePipelineOutput:
    metadata: list[list[MetaData]]
    embeddings: np.ndarray
    inputs: np.ndarray
    reconstructions: np.ndarray
    errors: np.ndarray
    model: Optional[Any] = None
