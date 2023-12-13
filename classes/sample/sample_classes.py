import sys

sys.path.append("..")

from dataclasses import dataclass
from typing import Any, Optional

from classes.common import Id
from classes.signal import Signal

from .sample_types import SampleTypes


@dataclass
class SingleSample:
    """Sample measurement containing data from a single cable"""

    x: Signal
    y: Signal
    z: Signal
    tension: float
    type: SampleTypes
    id: Id = Id()

    def __post_init__(self):
        assert self.x.type == self.y.type
        assert self.y.type == self.z.type
        self.type = self.x.type

    def to_dict(self) -> dict[str, Signal]:
        return {"x": self.x, "y": self.y, "z": self.z}

    def __getitem__(self, id: str) -> Signal:
        return self.to_dict()[id]

    def _get_items(self) -> list[Signal]:
        return [self.x, self.y, self.z]

    def compute_dct_threshold(self, thp: float):
        for x in self._get_items():
            x.compute_dct_threshold(thp)  # type: ignore

    def compress(self):
        assert self.type == SampleTypes.DCT
        for x in self._get_items():
            x.compress()

    def convert_to_dct(self):
        """Converts all signals to DCT."""
        if self.type != SampleTypes.DCT:
            self.x = self.x.to_dct()
            self.y = self.y.to_dct()
            self.z = self.z.to_dct()
            self.type = SampleTypes.DCT

    def convert_to_time(self):
        """Convert all signals to Time."""
        if self.type != SampleTypes.TIME:
            self.x = self.x.to_time()
            self.y = self.y.to_time()
            self.z = self.z.to_time()
            self.type = SampleTypes.TIME


@dataclass
class Sample:
    """Sample measurements containing data from all cables"""

    sample_id: str
    type: SampleTypes
    sample_c1: SingleSample
    sample_c2: SingleSample
    sample_c3: SingleSample
    sample_c4: SingleSample
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        assert self.sample_c1.type == self.sample_c2.type
        assert self.sample_c2.type == self.sample_c3.type
        assert self.sample_c3.type == self.sample_c4.type
        self.type = self.sample_c1.type

    def _get_items(self) -> list[SingleSample]:
        return [self.sample_c1, self.sample_c2, self.sample_c3, self.sample_c4]

    def __getitem__(self, idx: int) -> SingleSample:
        return self._get_items()[idx]

    def convert_to_dct(self):
        """Converts all SingleSamples to DCT."""
        if self.type != SampleTypes.DCT:
            for sample in self._get_items():
                sample.convert_to_dct()
            self.type = SampleTypes.DCT

    def convert_to_time(self):
        """Converts all SingleSamples to Time."""
        if self.type != SampleTypes.TIME:
            for sample in self._get_items():
                sample.convert_to_time()
            self.type = SampleTypes.TIME

    def compress(self, thp=0.1):
        assert self.type == SampleTypes.DCT

        for cable_sample in self._get_items():
            cable_sample.compute_dct_threshold(thp)
            cable_sample.compress()
