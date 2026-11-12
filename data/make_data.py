from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "fhvhv": ("fhvhv_tripdata_2026-09.parquet", "fhvhv_tripdata_2026-09.parquet"),
    "yellow": ("yellow_tripdata_2026-09.parquet", "yellow_tripdata_2026-09.parquet"),
    "green": ("green_tripdata_2026-09.parquet", "green_tripdata_2026-09.parquet"),
}
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def ensure_source_files(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for source, (filename, remote_name) in SOURCES.items():
        path = data_dir / filename
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            url = f"{BASE_URL}/{remote_name}"
            print(f"Downloading {url}")
            try:
                urllib.request.urlretrieve(url, partial)
                os.replace(partial, path)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        paths[source] = path
    return paths


def _read_zone_ids(path: Path) -> tuple[set[int], dict[int, str]]:
    zones = pd.read_csv(path)
    required = {"LocationID", "Zone"}
    missing = required.difference(zones.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    names = {int(row.LocationID): str(row.Zone) for row in zones.itertuples()}
    return set(names), names


def _normalized_batches(path: Path, source: str, batch_size: int):
    if source == "fhvhv":
        columns = ["pickup_datetime", "dropoff_datetime", "PULocationID",
                   "DOLocationID", "trip_miles", "trip_time"]
        pickup_name = "pickup_datetime"
        distance_name = "trip_miles"
    elif source == "yellow":
        columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                   "PULocationID", "DOLocationID", "trip_distance"]
        pickup_name = "tpep_pickup_datetime"
        distance_name = "trip_distance"
    else:
        columns = ["lpep_pickup_datetime", "lpep_dropoff_datetime",
                   "PULocationID", "DOLocationID", "trip_distance"]
        pickup_name = "lpep_pickup_datetime"
        distance_name = "trip_distance"

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        raw = batch.to_pydict()
        pickup = pd.Series(pd.to_datetime(raw[pickup_name], errors="coerce"))
        pu = pd.to_numeric(pd.Series(raw["PULocationID"]), errors="coerce")
        do = pd.to_numeric(pd.Series(raw["DOLocationID"]), errors="coerce")
        distance = pd.to_numeric(pd.Series(raw[distance_name]), errors="coerce")
        if source == "fhvhv":
            duration = pd.to_numeric(pd.Series(raw["trip_time"]), errors="coerce") / 60.0
        else:
            dropoff_name = ("tpep_dropoff_datetime" if source == "yellow"
                            else "lpep_dropoff_datetime")
            dropoff = pd.Series(pd.to_datetime(raw[dropoff_name], errors="coerce"))
            duration = (dropoff - pickup).dt.total_seconds() / 60.0

        valid = (
            pickup.notna()
            & pu.notna()
            & do.notna()
            & distance.notna()
            & duration.notna()
            & (pickup.dt.year == 2026)
            & (pickup.dt.month == 9)
            & (pickup.dt.day_of_week < 5)
            & duration.between(1, 120, inclusive="both")
            & (distance > 0)
        )
        if not valid.any():
            continue
        selected = np.flatnonzero(valid.to_numpy())
        frame = pd.DataFrame({
            "pickup_datetime": pickup.iloc[selected].to_numpy(),
            "pickup_zone": pu.iloc[selected].astype("int32").to_numpy(),
            "dropoff_zone": do.iloc[selected].astype("int32").to_numpy(),
            "trip_distance": distance.iloc[selected].astype("float32").to_numpy(),
            "duration_min": duration.iloc[selected].astype("float32").to_numpy(),
            "source_type": source,
        })
        yield frame


def _valid_zone_rows(frame: pd.DataFrame, location_ids: set[int]) -> pd.DataFrame:
    valid = (frame["pickup_zone"].isin(location_ids)
             & frame["dropoff_zone"].isin(location_ids))
    return frame.loc[valid].copy()


def _accumulate_training_stats(
    paths: dict[str, Path],
    location_ids: set[int],
    batch_size: int,
) -> tuple[dict[int, int], dict[int, float], dict[int, int], int]:
    pickup_counts: dict[int, int] = defaultdict(int)
    pair_distance_sums: dict[int, float] = defaultdict(float)
    pair_counts: dict[int, int] = defaultdict(int)
    total_rows = 0
    for source, path in paths.items():
        for frame in _normalized_batches(path, source, batch_size):
            frame = _valid_zone_rows(frame, location_ids)
            training = frame.loc[frame["pickup_datetime"].dt.day <= 23]
            if training.empty:
                continue
            total_rows += len(training)
            pu = training["pickup_zone"].to_numpy(dtype=np.int64)
            do = training["dropoff_zone"].to_numpy(dtype=np.int64)
            distances = training["trip_distance"].to_numpy(dtype=np.float64)
            counts = np.bincount(pu, minlength=266)
            for zone in np.flatnonzero(counts):
                pickup_counts[int(zone)] += int(counts[zone])
            keys, inverse = np.unique(pu * 1000 + do, return_inverse=True)
            sums = np.bincount(inverse, weights=distances)
            counts_by_pair = np.bincount(inverse)
            for key, distance_sum, count in zip(keys, sums, counts_by_pair):
                pair_distance_sums[int(key)] += float(distance_sum)
                pair_counts[int(key)] += int(count)
    return pickup_counts, pair_distance_sums, pair_counts, total_rows


def _reservoir_add(
    stored: pd.DataFrame | None,
    scores: np.ndarray,
    candidates: pd.DataFrame,
    limit: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame | None, np.ndarray]:
    if candidates.empty or limit <= 0:
        return stored, scores
    candidate_scores = rng.random(len(candidates))
    if stored is None:
        combined = candidates.reset_index(drop=True)
        combined_scores = candidate_scores
    else:
        combined = pd.concat([stored, candidates], ignore_index=True)
        combined_scores = np.concatenate((scores, candidate_scores))
    if len(combined) > limit:
        kept = np.argpartition(combined_scores, limit - 1)[:limit]
        combined = combined.iloc[kept].reset_index(drop=True)
        combined_scores = combined_scores[kept]
    return combined, combined_scores


def build_dataset(
    data_dir: Path = ROOT,
    output_dir: Path = ROOT / "data" / "processed",
    zone_lookup: Path | None = None,
    seed: int = 42,
    max_rows: int = 300_000,
    batch_size: int = 262_144,
) -> dict:
    if max_rows < 10:
        raise ValueError("max_rows must be at least 10")
    paths = ensure_source_files(data_dir)
    zone_lookup = zone_lookup or data_dir / "taxi_zone_lookup.csv"
    if not zone_lookup.exists():
        raise FileNotFoundError(f"Taxi-zone lookup not found: {zone_lookup}")
    location_ids, zone_names = _read_zone_ids(zone_lookup)

    pickup_counts, distance_sums, pair_counts, training_rows = (
        _accumulate_training_stats(paths, location_ids, batch_size)
    )
    top_zones = sorted(
        pickup_counts,
        key=lambda zone: (-pickup_counts[zone], zone),
    )[:20]
    if len(top_zones) < 2:
        raise ValueError("Fewer than two valid pickup zones found in the training data")

    distance_pairs = {
        key: distance_sums[key] / pair_counts[key]
        for key in pair_counts
    }
    relevant_pairs = {
        key: value for key, value in distance_pairs.items()
        if key // 1000 in top_zones and key % 1000 in top_zones
    }
    global_distance = float(
        sum(distance_sums.values()) / max(sum(pair_counts.values()), 1)
    )
    fallback_pairs = 0

    rng = np.random.default_rng(seed)
    train_limit = int(max_rows * 0.8)
    test_limit = max_rows - train_limit
    train_sample: pd.DataFrame | None = None
    test_sample: pd.DataFrame | None = None
    train_scores = np.empty(0)
    test_scores = np.empty(0)
    eligible_train = 0
    eligible_test = 0
    zone_array = np.asarray(top_zones, dtype=np.int32)

    for source, path in paths.items():
        for frame in _normalized_batches(path, source, batch_size):
            frame = _valid_zone_rows(frame, location_ids)
            keep = (frame["pickup_zone"].isin(zone_array)
                    & frame["dropoff_zone"].isin(zone_array))
            frame = frame.loc[keep].copy()
            if frame.empty:
                continue
            frame["hour"] = frame["pickup_datetime"].dt.hour.astype("int8")
            frame["weekday"] = frame["pickup_datetime"].dt.day_of_week.astype("int8")
            training = frame.loc[frame["pickup_datetime"].dt.day <= 23]
            testing = frame.loc[frame["pickup_datetime"].dt.day >= 24]
            eligible_train += len(training)
            eligible_test += len(testing)
            train_sample, train_scores = _reservoir_add(
                train_sample, train_scores, training, train_limit, rng
            )
            test_sample, test_scores = _reservoir_add(
                test_sample, test_scores, testing, test_limit, rng
            )

    if train_sample is None or test_sample is None or test_sample.empty:
        raise ValueError("No usable train/test records found after applying the date and zone filters")

    def add_distance_feature(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal fallback_pairs
        keys = (frame["pickup_zone"].to_numpy(dtype=np.int64) * 1000
                + frame["dropoff_zone"].to_numpy(dtype=np.int64))
        values = np.empty(len(keys), dtype=np.float32)
        for index, key in enumerate(keys):
            if int(key) in relevant_pairs:
                values[index] = relevant_pairs[int(key)]
            else:
                values[index] = global_distance
                fallback_pairs += 1
        frame["mean_pair_distance"] = values
        return frame

    train_sample = add_distance_feature(train_sample)
    test_sample = add_distance_feature(test_sample)
    columns = [
        "pickup_datetime", "pickup_zone", "dropoff_zone", "hour", "weekday",
        "mean_pair_distance", "trip_distance", "duration_min", "source_type",
    ]
    train_sample = train_sample[columns].sort_values("pickup_datetime").reset_index(drop=True)
    test_sample = test_sample[columns].sort_values("pickup_datetime").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_sample.to_parquet(train_path, index=False)
    test_sample.to_parquet(test_path, index=False)

    metadata = {
        "seed": seed,
        "max_sample_rows": max_rows,
        "source_files": {source: path.name for source, path in paths.items()},
        "source_rows": {
            source: pq.ParquetFile(path).metadata.num_rows for source, path in paths.items()
        },
        "valid_training_rows_before_zone_filter": training_rows,
        "eligible_training_rows": eligible_train,
        "eligible_test_rows": eligible_test,
        "train_sample_rows": len(train_sample),
        "test_sample_rows": len(test_sample),
        "training_period": "2026-09-01 through 2026-09-23, weekdays only",
        "test_period": "2026-09-24 through 2026-09-30, weekdays only",
        "top_20_pickup_zones": top_zones,
        "top_zone_names": {str(zone): zone_names[zone] for zone in top_zones},
        "pair_mean_distance_from_training_only": {
            f"{key // 1000}:{key % 1000}": value
            for key, value in relevant_pairs.items()
        },
        "global_mean_distance_from_training_only": global_distance,
        "distance_fallback_rows": fallback_pairs,
        "processed_train_file": train_path.name,
        "processed_test_file": test_path.name,
        "features": [
            "pickup_zone", "dropoff_zone", "hour", "weekday", "mean_pair_distance"
        ],
        "target": "duration_min",
        "filters": [
            "weekday pickups in September 2026",
            "pickup and drop-off IDs present in taxi_zone_lookup.csv",
            "trip distance greater than zero",
            "trip duration from 1 through 120 minutes",
            "both locations among the 20 most common training pickup zones",
        ],
        "data_origin": "Real NYC TLC Yellow, Green, and FHVHV trip records",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps({
        "train_rows": len(train_sample),
        "test_rows": len(test_sample),
        "top_zones": top_zones,
        "output_dir": str(output_dir),
    }, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real September 2026 TLC trip records.")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=262_144)
    args = parser.parse_args()
    build_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
