import sys

sys.path.append("..")

from pydantic import BaseModel
from ..signal.signal_classes import (
    SignalSpectrum,
    SignalFRF,
    WaveletPacket,
)
from ..feature.feature_classes import StatisticalFeatures


# -- Single classes --


class SingleFRFSample(BaseModel):
    x: SignalFRF
    y: SignalFRF
    z: SignalFRF
    tension: float


class SinglePSDSample(BaseModel):
    x: SignalSpectrum
    y: SignalSpectrum
    z: SignalSpectrum
    tension: float


class SingleWaveletPacketSample(BaseModel):
    x: WaveletPacket
    y: WaveletPacket
    z: WaveletPacket
    tension: float


# -- Sample classes --


class SampleFRF(BaseModel):
    sample_id: str
    sample_c1: SingleFRFSample
    sample_c2: SingleFRFSample
    sample_c3: SingleFRFSample
    sample_c4: SingleFRFSample


class SamplePSD(BaseModel):
    sample_id: str
    sample_c1: SinglePSDSample
    sample_c2: SinglePSDSample
    sample_c3: SinglePSDSample
    sample_c4: SinglePSDSample


class SampleWaveletPacket(BaseModel):
    sample_id: str
    sample_c1: SingleWaveletPacketSample
    sample_c2: SingleWaveletPacketSample
    sample_c3: SingleWaveletPacketSample
    sample_c4: SingleWaveletPacketSample


# -- Sample statistical features classes --


class SingleStatisticalFeaturesSample(BaseModel):
    x: StatisticalFeatures
    y: StatisticalFeatures
    z: StatisticalFeatures
    tension: float


class SampleStatisticalFeatures(BaseModel):
    sample_id: str
    temperature: float
    sample_c1: SingleStatisticalFeaturesSample
    sample_c2: SingleStatisticalFeaturesSample
    sample_c3: SingleStatisticalFeaturesSample
    sample_c4: SingleStatisticalFeaturesSample
