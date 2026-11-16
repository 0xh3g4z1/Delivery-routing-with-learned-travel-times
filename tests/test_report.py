import json
from pathlib import Path

import pandas as pd

from generate_report import generate_report


def test_report_generator_writes_sections_and_charts(tmp_path: Path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    experiments = tmp_path / "outputs" / "experiments"
    experiments.mkdir(parents=True)

    metrics = {
        "train_rows": 100,
        "test_rows": 25,
        "features": ["pickup_zone", "hour"],
        "target": "duration_min",
        "test_period": "Sep 24 through Sep 30",
        "nn": {"mae_minutes": 5.0, "rmse_minutes": 7.0},
        "historical_pair_average": {"mae_minutes": 6.0, "rmse_minutes": 8.0},
        "distance_only": {"mae_minutes": 10.0, "rmse_minutes": 12.0},
    }
    metadata = {
        "source_rows": {"yellow": 1000},
        "training_period": "Sep 1 through Sep 23",
    }
    (tmp_path / "artifacts" / "model_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (tmp_path / "data" / "processed" / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    pd.DataFrame([
        {"stops": 6, "algorithm": algorithm, "nodes_expanded_mean": nodes,
         "cost_minutes_mean": cost}
        for algorithm, nodes, cost in (
            ("astar", 10, 20), ("ucs", 12, 20), ("greedy", 5, 25),
        )
    ]).to_csv(experiments / "experiment1_summary.csv", index=False)
    pd.DataFrame([
        {"orders": 30, "configuration": label, "backtracks_mean": backtracks,
         "nodes_mean": nodes, "runtime_seconds_mean": runtime}
        for label, backtracks, nodes, runtime in (
            ("BT", 10, 20, 0.1), ("BT+FC", 0, 10, 0.01),
            ("BT+FC+AC-3", 0, 10, 0.02),
        )
    ]).to_csv(experiments / "experiment2_summary.csv", index=False)
    pd.DataFrame([
        {"model": model, "mae_minutes_mean": mae,
         "simulated_late_rate_mean": rate}
        for model, mae, rate in (
            ("NN", 5, 0.1), ("Historical average", 6, 0.2),
            ("Distance only", 10, 0.3),
        )
    ]).to_csv(experiments / "experiment3_summary.csv", index=False)
    pd.DataFrame([
        {"orders": 30, "configuration": "BT", "complete": False},
        {"orders": 30, "configuration": "BT+FC", "complete": True},
        {"orders": 30, "configuration": "BT+FC+AC-3", "complete": True},
    ]).to_csv(experiments / "experiment2_raw.csv", index=False)

    report_path = generate_report(tmp_path)

    report = report_path.read_text(encoding="utf-8")
    assert "## Abstract" in report
    assert "## 6. Conclusion" in report
    assert "not observed delivery lateness" in report
    assert "50.0%" in report
    for name in (
        "model_accuracy.png",
        "route_search.png",
        "csp_assignment.png",
        "replay_proxy.png",
    ):
        assert (tmp_path / "outputs" / "report_figures" / name).is_file()
        assert name in report
