from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import webbrowser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.make_data import build_dataset
from integration.planner import generate_orders, plan_orders
from integration.presentation import render_presentation
from integration.travel import TravelTimeOracle, export_hourly_matrix
from nn.model import load_bundle
from nn.train import train

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "artifacts" / "duration_model.joblib"
OUTPUTS = ROOT / "outputs"


def _ensure_model(seed: int) -> None:
    if not (PROCESSED / "train.parquet").exists():
        build_dataset(output_dir=PROCESSED, seed=seed)
    if not MODEL_PATH.exists():
        train(processed_dir=PROCESSED, artifact=MODEL_PATH, seed=seed, ensure_data=False)


def _plot_plan(orders, plan, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    flow = [
        ("REAL TLC TRIPS", "NN duration model\n+ baselines"),
        ("Predicted zone/hour times", "CSP assigns orders\n(time + capacity)"),
        ("Van-specific orders", "A* finds feasible\nstop order"),
    ]
    x_positions = [0.17, 0.5, 0.83]
    for (heading, detail), x in zip(flow, x_positions):
        axes[0].text(
            x, 0.55, f"{heading}\n\n{detail}", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e9f2fb",
                  "edgecolor": "#416788"},
            transform=axes[0].transAxes,
        )
    for x1, x2 in zip(x_positions, x_positions[1:]):
        axes[0].annotate(
            "", xy=(x2 - 0.13, 0.55), xytext=(x1 + 0.13, 0.55),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#416788"},
        )
    axes[0].set_title("How the three project parts cooperate")
    axes[0].axis("off")

    for row, (van_id, route_result) in enumerate(plan.routes.items()):
        for order in route_result.route or ():
            arrival = route_result.arrivals[order.order_id]
            axes[1].barh(
                row,
                order.window_end - order.window_start,
                left=order.window_start,
                height=0.55,
                color="#9ecae1",
                edgecolor="#3182bd",
            )
            axes[1].plot(arrival, row, marker="o", color="#d62728", markersize=4)
            axes[1].text(
                arrival + 2, row + 0.08, order.order_id,
                fontsize=6, rotation=45, ha="left",
            )
    axes[1].set_yticks(range(len(plan.routes)), list(plan.routes))
    axes[1].set_xlabel("Minutes after midnight (blue = allowed window, red dot = predicted arrival)")
    axes[1].set_title("A* route arrivals against generated order windows")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_xlim(8 * 60, 16 * 60)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_demo(
    seed: int = 42,
    ensure_assets: bool = True,
    time_model: str = "nn",
    scenario: str = "standard",
) -> dict:
    if scenario not in {"standard", "stress"}:
        raise ValueError(f"Unknown demo scenario: {scenario}")
    if ensure_assets:
        _ensure_model(seed)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}; run python -m nn.train first."
        )
    bundle = load_bundle(MODEL_PATH)
    oracle = TravelTimeOracle(bundle, method=time_model)
    oracle.precompute()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    matrix = export_hourly_matrix(
        oracle,
        OUTPUTS / f"predicted_zone_times_by_hour_{time_model}.csv",
        weekday=0,
    )
    zones = oracle.zones
    depot_zone = zones[0]
    if scenario == "stress":
        orders = generate_orders(
            11, zones, seed=seed, start_time=8 * 60,
            start_spread=30, window_width=(45, 120),
        )
    else:
        orders = generate_orders(
            20, zones, seed=seed, start_time=8 * 60,
            start_spread=60, window_width=(300, 420),
        )
    name_suffix = f"seed{seed}_{time_model}"
    if scenario == "standard":
        presentation_name = f"demo_presentation_{name_suffix}.html"
        plan_name = f"demo_plan_{name_suffix}.json"
        plot_name = f"demo_routes_{time_model}.png"
        order_name = f"demo_orders_seed{seed}.csv"
    else:
        presentation_name = (
            f"demo_presentation_{scenario}_{name_suffix}.html"
        )
        plan_name = f"demo_plan_{scenario}_{name_suffix}.json"
        plot_name = f"demo_routes_{scenario}_{name_suffix}.png"
        order_name = f"demo_orders_{scenario}_seed{seed}.csv"
    order_file = OUTPUTS / order_name
    pd.DataFrame([asdict(order) for order in orders]).to_csv(order_file, index=False)
    capacities = {"Van 1": 10, "Van 2": 10, "Van 3": 10}
    plan = plan_orders(
        orders, capacities, depot_zone, 8 * 60, oracle, weekday=0,
    )
    presentation_path = OUTPUTS / presentation_name
    presentation_orders = [
        {
            "order_id": order.order_id,
            "zone": order.zone,
            "zone_name": oracle.zone_names.get(order.zone, "Unknown"),
            "window_start": order.window_start,
            "window_end": order.window_end,
            "demand": order.demand,
            "service_minutes": order.service_minutes,
        }
        for order in orders
    ]
    if not plan.feasible:
        plan_file = OUTPUTS / plan_name
        with plan_file.open("w", encoding="utf-8") as file:
            json.dump({
                "feasible": False,
                "reason": plan.reason,
                "window_suggestions": list(plan.window_suggestions),
                "time_model": time_model,
                "scenario": scenario,
                "planning_attempts": [asdict(item) for item in plan.attempts],
            }, file, indent=2)
        presentation_path.write_text(
            render_presentation({
                "seed": seed,
                "scenario": scenario,
                "time_model": time_model,
                "metrics": bundle["metadata"]["metrics"],
                "orders": presentation_orders,
                "depot_zone": depot_zone,
                "depot_name": oracle.zone_names.get(depot_zone, "Unknown"),
                "vans": [
                    {
                        "van_id": van_id,
                        "capacity": capacity,
                        "demand": 0,
                        "order_ids": [],
                        "route": [],
                        "cost": 0.0,
                        "expanded": 0,
                        "generated": 0,
                    }
                    for van_id, capacity in capacities.items()
                ],
                "csp_stats": asdict(plan.csp_stats),
                "planning_attempts": [asdict(item) for item in plan.attempts],
                "reassignments": plan.reassignments,
                "feasible": False,
                "reason": plan.reason,
                "suggestions": list(plan.window_suggestions),
            }),
            encoding="utf-8",
        )
        return {
            "feasible": False,
            "reason": plan.reason,
            "matrix_rows": len(matrix),
            "window_suggestions": list(plan.window_suggestions),
            "orders_file": str(order_file),
            "plan_file": str(plan_file),
            "presentation": str(presentation_path),
        }

    plot_path = OUTPUTS / plot_name
    _plot_plan(orders, plan, plot_path)
    assignment_counts = Counter(plan.assignments.values())
    vans = []
    orders_by_id = {order.order_id: order for order in orders}
    for van_id, capacity in capacities.items():
        assigned_order_ids = [
            order_id for order_id, assigned_van in plan.assignments.items()
            if assigned_van == van_id
        ]
        route_result = plan.routes.get(van_id)
        route = []
        if route_result is not None:
            route = [
                {
                    "order_id": order.order_id,
                    "window_start": order.window_start,
                    "window_end": order.window_end,
                    "arrival": route_result.arrivals[order.order_id],
                }
                for order in route_result.route or ()
            ]
        vans.append({
            "van_id": van_id,
            "capacity": capacity,
            "demand": sum(orders_by_id[order_id].demand
                          for order_id in assigned_order_ids),
            "order_ids": assigned_order_ids,
            "route": route,
            "cost": route_result.cost if route_result is not None else 0.0,
            "expanded": route_result.expanded if route_result is not None else 0,
            "generated": route_result.generated if route_result is not None else 0,
        })
    presentation_path.write_text(
        render_presentation({
            "seed": seed,
            "scenario": scenario,
            "time_model": time_model,
            "metrics": bundle["metadata"]["metrics"],
            "orders": presentation_orders,
            "depot_zone": depot_zone,
            "depot_name": oracle.zone_names.get(depot_zone, "Unknown"),
            "vans": vans,
            "csp_stats": asdict(plan.csp_stats),
            "planning_attempts": [asdict(item) for item in plan.attempts],
            "reassignments": plan.reassignments,
            "feasible": True,
            "reason": None,
            "suggestions": list(plan.window_suggestions),
        }),
        encoding="utf-8",
    )
    plan_file = OUTPUTS / plan_name
    with plan_file.open("w", encoding="utf-8") as file:
        json.dump({
            "feasible": True,
            "seed": seed,
            "scenario": scenario,
            "time_model": time_model,
            "depot_zone": depot_zone,
            "capacities": capacities,
            "assignments": plan.assignments,
            "routes": {
                van_id: {
                    "order_ids": [order.order_id for order in result.route or ()],
                    "arrivals_minutes_after_midnight": result.arrivals,
                    "cost_minutes": result.cost,
                    "nodes_expanded": result.expanded,
                    "nodes_generated": result.generated,
                }
                for van_id, result in plan.routes.items()
            },
            "csp_stats": asdict(plan.csp_stats),
            "planning_attempts": [asdict(item) for item in plan.attempts],
            "reassignments": plan.reassignments,
            "synthetic_orders": True,
        }, file, indent=2)
    summary = {
        "feasible": True,
        "orders": len(orders),
        "synthetic_orders": all(order.synthetic for order in orders),
        "real_data_train_rows": bundle["metadata"]["metrics"]["train_rows"],
        "real_data_test_rows": bundle["metadata"]["metrics"]["test_rows"],
        "matrix_rows": len(matrix),
        "assigned_per_van": dict(assignment_counts),
        "reassignments": plan.reassignments,
        "time_model": time_model,
        "scenario": scenario,
        "orders_file": str(order_file),
        "plan_file": str(plan_file),
        "plot": str(plot_path),
        "presentation": str(presentation_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TLC learned-time routing demo.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to 42 for standard or 94 for stress).",
    )
    parser.add_argument("--time-model", choices=("nn", "historical", "distance"),
                        default="nn")
    parser.add_argument(
        "--scenario",
        choices=("standard", "stress"),
        default="standard",
        help="Use the normal demo or a tighter-window case exercising replanning.",
    )
    parser.add_argument("--no-prepare", action="store_true",
                        help="Fail rather than build data/model assets that are missing.")
    parser.add_argument("--no-open", action="store_true",
                        help="Generate the presentation without opening a browser.")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (
        94 if args.scenario == "stress" else 42
    )
    summary = run_demo(
        seed=seed,
        ensure_assets=not args.no_prepare,
        time_model=args.time_model,
        scenario=args.scenario,
    )
    presentation_path = Path(summary["presentation"]).resolve()
    print(f"Presentation generated: {presentation_path}")
    if not args.no_open:
        webbrowser.open(presentation_path.as_uri())


if __name__ == "__main__":
    main()
