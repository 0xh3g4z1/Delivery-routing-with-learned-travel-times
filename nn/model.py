from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "weekday"]
NUMERIC_FEATURES = ["mean_pair_distance"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "duration_min"


def build_model(seed: int = 42, max_iter: int = 100) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("distance", StandardScaler(), NUMERIC_FEATURES),
        ],
        sparse_threshold=0.3,
    )
    network = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        batch_size=1024,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        tol=0.001,
        validation_fraction=0.1,
        random_state=seed,
    )
    return Pipeline([("features", transformer), ("network", network)])


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


def score_predictions(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    errors = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return Metrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
    )


class HistoricalPairAverage:
    def __init__(self) -> None:
        self.pair_means: dict[tuple[int, int], float] = {}
        self.global_mean = 0.0

    def fit(self, train: pd.DataFrame) -> HistoricalPairAverage:
        self.global_mean = float(train[TARGET].mean())
        means = train.groupby(["pickup_zone", "dropoff_zone"])[TARGET].mean()
        self.pair_means = {
            (int(pair[0]), int(pair[1])): float(value)
            for pair, value in means.items()
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([
            self.pair_means.get((int(pu), int(do)), self.global_mean)
            for pu, do in zip(frame["pickup_zone"], frame["dropoff_zone"])
        ])


class DistanceOnlyAverage:
    def __init__(self) -> None:
        self.minutes_per_mile = 0.0

    def fit(self, train: pd.DataFrame) -> DistanceOnlyAverage:
        ratios = train.loc[
            train["trip_distance"] > 0,
            TARGET,
        ].to_numpy(dtype=float) / train.loc[
            train["trip_distance"] > 0, "trip_distance"
        ].to_numpy(dtype=float)
        self.minutes_per_mile = float(np.median(ratios))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(
            1.0,
            frame["trip_distance"].to_numpy(dtype=float) * self.minutes_per_mile,
        )


def save_bundle(
    path: Path,
    model: Pipeline,
    pair_average: HistoricalPairAverage,
    distance_average: DistanceOnlyAverage,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "pair_average": pair_average,
        "distance_average": distance_average,
        "metadata": metadata,
    }, path)


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "pair_average", "distance_average", "metadata"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle
