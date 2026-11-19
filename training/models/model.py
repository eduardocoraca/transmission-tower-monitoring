import sys

sys.path.append("..")

from classes.sample.sample_classes import SampleWaveletPacket
from typing import Literal, cast
from pydantic import BaseModel
from datetime import datetime
import numpy as np
from lib.wpt_utils import combine_wpts, build_index_per_level
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

CABLES = ["c1", "c2", "c3", "c4"]
AXES = ["x", "y", "z"]


# -- Hyperparams --


class FeatureParams(BaseModel):
    f_min: float
    f_max: float
    fs: float
    max_level: int
    db: bool


class ReductionParams(BaseModel):
    n_components: int


class SequenceParams(BaseModel):
    n_states: int


# -- Objects --


class BandRange(BaseModel):
    start: float
    end: float

    @property
    def bandwidth(self) -> float:
        return self.end - self.start

    @property
    def id(self) -> str:
        return f"{self.start}-{self.end}"


class SpectralFeatures(BaseModel):
    sampled_at: datetime
    id: str
    tension: float
    cable: Literal["c1", "c2", "c3", "c4"]
    value: float
    level: int | None = None


class ModelCalibration(BaseModel):
    cable: Literal["c1", "c2", "c3", "c4"]
    id: str
    value: float


# -- Utils --


def has_intersection(range_0: BandRange, range_1: BandRange) -> bool:
    if range_1.start >= range_0.end:
        return False

    if range_0.start >= range_1.end:
        return False

    if range_1.start > range_0.start and range_0.end <= range_1.end:
        return True

    if range_0.start > range_1.start and range_1.end <= range_0.end:
        return True

    if range_1.start > range_0.start and range_1.end <= range_0.end:
        return True

    if range_0.start > range_1.start and range_0.end <= range_1.end:
        return True

    return False


def features_to_df(features: list[SpectralFeatures]) -> pd.DataFrame:
    df = pd.DataFrame([f.model_dump() for f in features])
    df["key"] = df["cable"].astype(str) + "_" + df["sampled_at"].astype(str)
    return df


# -- Feature Extractor --


class FeatureExtractor:
    def __init__(self, params: FeatureParams):
        self.params = params

    def _extract_bands(self, sample_wpt: SampleWaveletPacket) -> list[SpectralFeatures]:
        """Extract spectral features from wavelet packet decomposition.

        Args:
            sample_wpt (SampleWaveletPacket): Wavelet packet decomposition for a sample
            f_min (float): Minimum frequency to consider (Hz)
            f_max (float): Maximum frequency to consider (Hz)
            fs (float): Sampling frequency (Hz)
            max_level (int): Maximum decomposition level

        Returns:
            list[SpectralFeatures]: List of spectral features containing:
                - energy: Raw energy in the frequency band
                - norm_energy: Energy normalized by maximum energy in the level
                - range: Frequency range of the band
                - level: Decomposition level
                - cable: Cable identifier
        """
        sampled_at = datetime.fromisoformat(sample_wpt.sample_id)
        allowed_range = BandRange(start=self.params.f_min, end=self.params.f_max)
        index_per_level = build_index_per_level(self.params.max_level + 1)
        features = []
        for cable in CABLES:
            wpt = getattr(sample_wpt, f"sample_{cable}")
            tension = wpt.tension
            combined_wpt = combine_wpts([wpt.y, wpt.z])
            raw_energy = np.array(combined_wpt.energy_map)

            for level, indexes in index_per_level.items():
                num_indexes = len(indexes)
                bw = (self.params.fs / 2) / num_indexes
                freq_ranges = [
                    BandRange(start=k * bw, end=(k + 1) * bw)
                    for k in range(num_indexes)
                ]
                level_energy = raw_energy[indexes]

                # select relevant subbands
                level_features = []
                max_level_energy = 0
                for band_range, raw in zip(freq_ranges, level_energy):
                    if has_intersection(band_range, allowed_range):
                        if raw > max_level_energy:
                            max_level_energy = raw
                        feat = SpectralFeatures(
                            value=raw / bw,  # normalized energy
                            id=band_range.id,
                            level=level,
                            cable=cast(Literal["c1", "c2", "c3", "c4"], cable),
                            sampled_at=sampled_at,
                            tension=tension,
                        )
                        level_features.append(feat)

                # discard levels with only one subband
                if len(level_features) == 1:
                    continue

                features += level_features
        return features

    def _extract_features(
        self, raw_features: list[SpectralFeatures]
    ) -> list[SpectralFeatures]:
        band_df = features_to_df(raw_features)
        pivoted_band_df = band_df.pivot(
            index=["cable", "sampled_at", "tension"], columns="id", values="value"
        )
        features = []
        for key, row in pivoted_band_df.iterrows():
            x = row.to_numpy()
            ratios = np.divide.outer(x, x)
            ind = np.triu_indices(len(ratios))
            triu_ratios = ratios[ind].ravel()
            if self.params.db:
                triu_ratios = 20 * np.log10(triu_ratios)
            for i, data in enumerate(triu_ratios):
                features.append(
                    SpectralFeatures(
                        sampled_at=key[1],
                        cable=key[0],
                        tension=key[2],
                        value=data,
                        id=str(i),
                    )
                )
        return features

    def transform(self, data: list[SampleWaveletPacket]) -> list[SpectralFeatures]:
        band_features: list[SpectralFeatures] = []
        for d in data:
            band_features += self._extract_bands(d)
        features = self._extract_features(band_features)
        return features


# -- Models --


class ReductionModel:
    def __init__(self, params: ReductionParams):
        self.params = params
        self.reduction_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.params.n_components)),
            ]
        )

    def fit(self, data: list[SpectralFeatures]) -> None:
        df = features_to_df(data)
        pivoted_df = df.pivot(
            index=["cable", "sampled_at", "tension"], columns="id", values="value"
        )
        x = pivoted_df.to_numpy()
        self.reduction_pipeline.fit(x)

    def transform(self, data: list[SpectralFeatures]) -> list[SpectralFeatures]:
        df = features_to_df(data)
        pivoted_df = df.pivot(
            index=["cable", "sampled_at", "tension"], columns="id", values="value"
        )
        x = pivoted_df.to_numpy()
        meta = pivoted_df.reset_index()[["cable", "sampled_at", "tension"]]
        y = self.reduction_pipeline.transform(x)
        out = []
        for record in zip(
            y,
            meta["sampled_at"].to_list(),
            meta["cable"].to_list(),
            meta["tension"].to_list(),
        ):
            for i, value in enumerate(record[0]):
                out.append(
                    SpectralFeatures(
                        sampled_at=record[1],
                        id=str(i),
                        tension=record[3],
                        value=value,
                        cable=record[2],
                    )
                )
        return out


class CalibrationModel:
    def __init__(self):
        self.calibrations: list[ModelCalibration] = []

    def calibrate(
        self,
        data: list[SpectralFeatures],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> None:
        features = features_to_df(data)
        if start_date is not None:
            features = features[features["sampled_at"] >= start_date]
        if end_date is not None:
            features = features[features["sampled_at"] <= end_date]
        grouped = features.groupby(["cable", "id"])["value"].mean().reset_index()
        self.calibrations = [ModelCalibration(**row) for _, row in grouped.iterrows()]

    def transform(self, data: list[SpectralFeatures]) -> list[SpectralFeatures]:
        transformed = []
        calibration_dict = {}
        for calibration in self.calibrations:
            calibration_dict[calibration.id, calibration.cable] = calibration.value

        for feature in data:
            feature.value = feature.value - calibration_dict[feature.id, feature.cable]
            transformed.append(feature)
        return transformed


class SequenceModel:
    def __init__(self, params: SequenceParams):
        self.params = params

    def _build_left_right_transition_matrix(self) -> np.ndarray:
        transition_matrix = np.zeros((self.params.n_states, self.params.n_states))
        for i in range(self.params.n_states):
            if i == self.params.n_states - 1:
                transition_matrix[i, i] = 1
            elif i == 0:
                transition_matrix[i, i] = 0.9
                transition_matrix[i, i + 1] = 0.1
            else:
                transition_matrix[i, i] = 0.9
                transition_matrix[i, i + 1] = 0.1
                # transition_matrix[i, i-1] = 0.1
        return transition_matrix

    def fit(self, data: list[SpectralFeatures]) -> None:
        df = features_to_df(data)
        cables = df.cable.unique()

        # Format data to hmmlearn format
        data_list: list[np.ndarray] = []
        lens: list[int] = []
        for cable in cables:
            cable_df = df.loc[df["cable"] == cable]
            cable_df = cable_df.sort_values(by="sampled_at")
            pivoted_df = cable_df.pivot(
                index=["sampled_at"], columns="id", values="value"
            )
            data_list.append(pivoted_df.to_numpy())
            lens.append(len(pivoted_df))
        data_list = np.vstack(data_list)

        # Fit model
        self.model = GaussianHMM(
            n_components=self.params.n_states,
            covariance_type="full",
            init_params="smc",
            params="stmc",
        )
        self.model.transmat_ = self._build_left_right_transition_matrix()
        self.model.fit(data_list, lens)

    def predict(
        self, data: list[SpectralFeatures]
    ) -> tuple[list[SpectralFeatures], list[float]]:
        df = features_to_df(data)
        cables = df.cable.unique()

        # Format data to hmmlearn format
        output: list[SpectralFeatures] = []
        scores: list[float] = []
        for cable in cables:
            cable_df = df.loc[df["cable"] == cable]
            cable_df = cable_df.sort_values(by="sampled_at")
            pivoted_df = cable_df.pivot(
                index=["sampled_at"], columns="id", values="value"
            )
            x = pivoted_df.to_numpy()
            y = self.model.predict(x)
            for i, value in enumerate(y):
                output.append(
                    SpectralFeatures(
                        sampled_at=cable_df.iloc[i]["sampled_at"],
                        id="0",
                        cable=cable,
                        value=value,
                        tension=cable_df.iloc[i]["tension"],
                    )
                )
            scores.append(self.model.score(x))
        return output, scores
