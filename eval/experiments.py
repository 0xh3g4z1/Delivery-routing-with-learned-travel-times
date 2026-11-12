from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csp.assignment import AssignmentSolver
from integration.planner import generate_orders
from integration.travel import TravelTimeOracle
from nn.model import load_bundle
from nn.train import train
from search.route import route_search

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "experiments"
PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "artifacts" / "duration_model.joblib"
SEEDS = (11, 23, 37, 51, 79)


def _ensure_model() -> dict:
    if not MODEL_PATH.exists():
        train(processed_dir=PROCESSED, artifact=MODEL_PATH)
    return load_bundle(MODEL_PATH)


def _search_experiment(oracle: TravelTimeOracle) -> pd.DataFrame:
    rows = []
    for count in range(6, 11):
        for seed in SEEDS:
            orders = generate_orders(
                count, oracle.zones, seed=seed, start_time=8 * 60,
                start_spread=60, window_width=(360, 480),
            )
            for algorithm in ("astar", "ucs", "greedy"):
                started = time.perf_counter()
                result = route_search(
                    orders, oracle.zones[0], 8 * 60, oracle,
                    algorithm=algorithm, weekday=seed % 5,
                    max_expansions=50_000,
                )
                rows.append({
                    "stops": count,
                    "seed": seed,
                    "algorithm": algorithm,
                    "cost_minutes": result.cost,
                    "nodes_expanded": result.expanded,
                    "nodes_generated": result.generated,
                    "complete": result.complete,
                    "feasible": result.feasible,
                    "runtime_seconds": time.perf_counter() - started,
                })
    return pd.DataFrame(rows)


def _csp_experiment(oracle: TravelTimeOracle) -> pd.DataFrame:
    rows = []
    for count in (15, 20, 30):
        for seed in SEEDS:
            orders = generate_orders(
                count, oracle.zones, seed=seed + 100, start_time=8 * 60,
                start_spread=60, window_width=(60, 120),
            )
            total_demand = sum(order.demand for order in orders)
            capacity = max(8, int(np.ceil(total_demand / 3)))
            capacities = {
                "Van 1": capacity - 2,
                "Van 2": capacity,
                "Van 3": capacity + 2,
            }
            for label, forward_checking, arc_consistency in (
                ("BT", False, False),
                ("BT+FC", True, False),
                ("BT+FC+AC-3", True, True),
            ):
                solver = AssignmentSolver(
                    orders, capacities, oracle.zones[0], 8 * 60,
                    oracle, weekday=seed % 5,
                )
                edge_count = len(solver.conflicts)
                started = time.perf_counter()
                result = solver.solve(
                    forward_checking=forward_checking,
                    arc_consistency=arc_consistency,
                    max_nodes=100_000,
                )
                rows.append({
                    "orders": count,
                    "seed": seed,
                    "configuration": label,
                    "conflict_edges": edge_count,
                    "assignments": result.stats.assignments,
                    "backtracks": result.stats.backtracks,
                    "nodes": result.stats.nodes,
                    "values_pruned": result.stats.values_pruned,
                    "feasible": result.feasible,
                    "complete": result.complete,
                    "runtime_seconds": time.perf_counter() - started,
                })
    return pd.DataFrame(rows)


def _replay_experiment(bundle: dict) -> pd.DataFrame:
    test = pd.read_parquet(PROCESSED / "test.parquet")
    pair_average = bundle["pair_average"]
    distance_average = bundle["distance_average"]
    model = bundle["model"]
    rng_buffer = 5.0
    rows = []
    sample_size = min(10_000, len(test))
    for seed in SEEDS:
        sample = test.sample(n=sample_size, random_state=seed)
        actual = sample["duration_min"].to_numpy(dtype=float)
        predictions = {
            "NN": model.predict(sample[[
                "pickup_zone", "dropoff_zone", "hour", "weekday",
                "mean_pair_distance",
            ]]),
            "Historical average": pair_average.predict(sample),
            "Distance only": distance_average.predict(sample),
        }
        due_budget = actual + rng_buffer
        for method, predicted in predictions.items():
            rows.append({
                "seed": seed,
                "model": method,
                "trips_replayed": sample_size,
                "simulated_late_count": int(np.sum(predicted > due_budget)),
                "simulated_late_rate": float(np.mean(predicted > due_budget)),
                "mae_minutes": float(np.mean(np.abs(predicted - actual))),
                "synthetic_slack_minutes": rng_buffer,
            })
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    summary = frame.groupby(keys)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def _write_plots(
    search_raw: pd.DataFrame,
    csp_raw: pd.DataFrame,
    replay_raw: pd.DataFrame,
) -> None:
    search_summary = search_raw.groupby(["stops", "algorithm"], as_index=False)[
        "nodes_expanded"
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    for algorithm, group in search_summary.groupby("algorithm"):
        ax.plot(group["stops"], group["nodes_expanded"], marker="o", label=algorithm)
    ax.set(title="Experiment 1: Mean route-search expansions (5 seeds)",
           xlabel="Stops", ylabel="Mean nodes expanded")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "experiment1_search.png", dpi=160)
    plt.close(fig)

    csp_summary = csp_raw.groupby(
        ["orders", "configuration"], as_index=False
    )["backtracks"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    for configuration, group in csp_summary.groupby("configuration"):
        ax.plot(group["orders"], group["backtracks"], marker="o", label=configuration)
    ax.set(title="Experiment 2: Mean CSP backtracks (5 seeds)",
           xlabel="Orders", ylabel="Mean backtracks")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "experiment2_csp.png", dpi=160)
    plt.close(fig)

    replay_summary = replay_raw.groupby("model", as_index=False)[
        "simulated_late_rate"
    ].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(replay_summary["model"], replay_summary["simulated_late_rate"])
    ax.set(title="Experiment 3: Simulated late-rate proxy (5 test samples)",
           ylabel="Mean proportion over the observed duration + 5 min")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "experiment3_replay.png", dpi=160)
    plt.close(fig)


def run_experiments(output_dir: Path = OUTPUT) -> dict[str, pd.DataFrame]:
    global OUTPUT
    OUTPUT = output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bundle = _ensure_model()
    oracle = TravelTimeOracle(bundle)
    oracle.precompute()
    search_raw = _search_experiment(oracle)
    csp_raw = _csp_experiment(oracle)
    replay_raw = _replay_experiment(bundle)

    search_summary = _summary(
        search_raw, ["stops", "algorithm"],
        ["nodes_expanded", "nodes_generated", "cost_minutes", "runtime_seconds"],
    )
    csp_summary = _summary(
        csp_raw, ["orders", "configuration"],
        [
            "assignments", "backtracks", "nodes", "values_pruned",
            "runtime_seconds",
        ],
    )
    replay_summary = _summary(
        replay_raw, ["model"],
        ["simulated_late_count", "simulated_late_rate", "mae_minutes"],
    )
    outputs = {
        "experiment1_raw": search_raw,
        "experiment1_summary": search_summary,
        "experiment2_raw": csp_raw,
        "experiment2_summary": csp_summary,
        "experiment3_raw": replay_raw,
        "experiment3_summary": replay_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    _write_plots(search_raw, csp_raw, replay_raw)
    for name, frame in (
        ("Experiment 1 summary", search_summary),
        ("Experiment 2 summary", csp_summary),
        ("Experiment 3 summary", replay_summary),
    ):
        print(f"\n{name}\n{frame.to_string(index=False)}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three required experiments.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run_experiments(args.output_dir)


if __name__ == "__main__":
    main()
