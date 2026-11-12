from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nn.model import FEATURES


class TravelTimeOracle:
    def __init__(
        self,
        bundle: dict,
        zones: list[int] | None = None,
        method: str = "nn",
    ) -> None:
        if method not in {"nn", "historical", "distance"}:
            raise ValueError(f"Unknown travel-time method: {method}")
        self.model = bundle["model"]
        self.pair_average = bundle["pair_average"]
        self.distance_average = bundle["distance_average"]
        self.method = method
        data_metadata = bundle["metadata"]["data"]
        self.zones = list(zones or data_metadata["top_20_pickup_zones"])
        self.zone_names = {
            int(zone): name for zone, name in data_metadata["top_zone_names"].items()
        }
        self.global_distance = float(
            data_metadata["global_mean_distance_from_training_only"]
        )
        self.pair_distances = {
            tuple(int(part) for part in key.split(":")): float(value)
            for key, value in data_metadata[
                "pair_mean_distance_from_training_only"
            ].items()
        }
        self.cache: dict[tuple[int, int, int, int], float] = {}

    def _distance(self, pickup_zone: int, dropoff_zone: int) -> float:
        return self.pair_distances.get(
            (pickup_zone, dropoff_zone), self.global_distance
        )

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.method == "nn":
            values = self.model.predict(frame[FEATURES])
        elif self.method == "historical":
            values = self.pair_average.predict(frame)
        else:
            values = (
                frame["mean_pair_distance"].to_numpy(dtype=float)
                * self.distance_average.minutes_per_mile
            )
        return np.clip(values, 1.0, 120.0)

    def precompute(self, weekdays: range = range(5), batch_size: int = 20_000) -> None:
        rows = [
            (pickup, dropoff, hour, weekday)
            for weekday in weekdays
            for hour in range(24)
            for pickup in self.zones
            for dropoff in self.zones
        ]
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            frame = pd.DataFrame(batch, columns=[
                "pickup_zone", "dropoff_zone", "hour", "weekday"
            ])
            frame["mean_pair_distance"] = [
                self._distance(int(pu), int(do))
                for pu, do in zip(frame["pickup_zone"], frame["dropoff_zone"])
            ]
            predictions = self._predict(frame)
            for row, prediction in zip(batch, predictions):
                pickup, dropoff, hour, weekday = row
                self.cache[(pickup, dropoff, hour, weekday)] = float(prediction)

    def __call__(
        self,
        pickup_zone: int,
        dropoff_zone: int,
        hour: float,
        weekday: int,
    ) -> float:
        key = (
            int(pickup_zone),
            int(dropoff_zone),
            int(hour) % 24,
            int(weekday) % 7,
        )
        if key not in self.cache:
            row = pd.DataFrame([{
                "pickup_zone": key[0],
                "dropoff_zone": key[1],
                "hour": key[2],
                "weekday": key[3],
                "mean_pair_distance": self._distance(key[0], key[1]),
            }])
            self.cache[key] = float(self._predict(row)[0])
        return self.cache[key]

    def hourly_frame(self, weekday: int = 0) -> pd.DataFrame:
        if not all(
            (pickup, dropoff, hour, weekday) in self.cache
            for pickup in self.zones
            for dropoff in self.zones
            for hour in range(24)
        ):
            raise RuntimeError("Precompute travel predictions before exporting a matrix")
        rows = []
        for pickup in self.zones:
            for dropoff in self.zones:
                for hour in range(24):
                    rows.append({
                        "pickup_zone": pickup,
                        "pickup_name": self.zone_names.get(pickup, "Unknown"),
                        "dropoff_zone": dropoff,
                        "dropoff_name": self.zone_names.get(dropoff, "Unknown"),
                        "hour": hour,
                        "weekday": weekday,
                        "predicted_minutes": self(pickup, dropoff, hour, weekday),
                    })
        return pd.DataFrame(rows)


def export_hourly_matrix(
    oracle: TravelTimeOracle,
    path: Path,
    weekday: int = 0,
) -> pd.DataFrame:
    frame = oracle.hourly_frame(weekday)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
