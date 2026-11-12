from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.make_data import build_dataset
from nn.model import (
    FEATURES,
    TARGET,
    DistanceOnlyAverage,
    HistoricalPairAverage,
    build_model,
    save_bundle,
    score_predictions,
)

def train(
    processed_dir: Path = ROOT / "data" / "processed",
    artifact: Path = ROOT / "artifacts" / "duration_model.joblib",
    seed: int = 42,
    max_iter: int = 100,
    ensure_data: bool = True,
) -> dict:
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    metadata_path = processed_dir / "metadata.json"
    if not (train_path.exists() and test_path.exists() and metadata_path.exists()):
        if not ensure_data:
            raise FileNotFoundError(
                f"Processed data missing under {processed_dir}; run python data/make_data.py"
            )
        build_dataset(output_dir=processed_dir, seed=seed)

    train_frame = pd.read_parquet(train_path)
    test_frame = pd.read_parquet(test_path)
    with metadata_path.open(encoding="utf-8") as file:
        data_metadata = json.load(file)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Training and test files must both contain real trips")
    missing = set(FEATURES + [TARGET]).difference(train_frame.columns)
    if missing:
        raise ValueError(f"Processed training data is missing columns: {sorted(missing)}")

    model = build_model(seed=seed, max_iter=max_iter)
    model.fit(train_frame[FEATURES], train_frame[TARGET])
    predictions = model.predict(test_frame[FEATURES])
    neural_metrics = score_predictions(test_frame[TARGET].to_numpy(), predictions)

    pair_average = HistoricalPairAverage().fit(train_frame)
    pair_predictions = pair_average.predict(test_frame)
    pair_metrics = score_predictions(test_frame[TARGET].to_numpy(), pair_predictions)

    distance_average = DistanceOnlyAverage().fit(train_frame)
    distance_predictions = distance_average.predict(test_frame)
    distance_metrics = score_predictions(test_frame[TARGET].to_numpy(), distance_predictions)

    metrics = {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "seed": seed,
        "nn": {"mae_minutes": neural_metrics.mae, "rmse_minutes": neural_metrics.rmse},
        "historical_pair_average": {
            "mae_minutes": pair_metrics.mae,
            "rmse_minutes": pair_metrics.rmse,
        },
        "distance_only": {
            "mae_minutes": distance_metrics.mae,
            "rmse_minutes": distance_metrics.rmse,
        },
        "neural_network_iterations": int(model.named_steps["network"].n_iter_),
        "test_period": data_metadata["test_period"],
        "target": TARGET,
        "features": FEATURES,
        "all_data_rows_are_real_trips": True,
    }
    artifact_metadata = {
        "data": data_metadata,
        "metrics": metrics,
        "seed": seed,
        "max_iter": max_iter,
    }
    save_bundle(artifact, model, pair_average, distance_average, artifact_metadata)
    metrics_path = artifact.parent / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train duration NN and real-data baselines.")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "artifacts" / "duration_model.joblib")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    train(args.processed_dir, args.artifact, args.seed, args.max_iter)


if __name__ == "__main__":
    main()
