from __future__ import annotations

from pathlib import Path

from csp.assignment import AssignmentSolver
from integration.planner import generate_orders
from integration.travel import TravelTimeOracle
from integration.models import Order
from nn.model import load_bundle
from nn.train import train
from search.route import brute_force_route, route_search

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "duration_model.joblib"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    if not MODEL_PATH.exists():
        train(processed_dir=PROCESSED, artifact=MODEL_PATH)
    bundle = load_bundle(MODEL_PATH)
    oracle = TravelTimeOracle(bundle)
    oracle.precompute()

    route_orders = generate_orders(
        6, oracle.zones, seed=13, start_time=8 * 60,
        start_spread=30, window_width=(360, 420),
    )
    route = route_search(
        route_orders, oracle.zones[0], 8 * 60, oracle,
        algorithm="astar", weekday=0,
    )
    brute = brute_force_route(route_orders, oracle.zones[0], 8 * 60, oracle)
    print("A* trace comparison (6 generated stops, real learned travel-time model):")
    print(f"  A*: feasible={route.feasible}, cost={route.cost}, "
          f"expanded={route.expanded}, generated={route.generated}")
    print(f"  Brute-force best cost={None if brute is None else brute[0]}")
    print("  First search states:")
    for line in route.trace[:8]:
        print(f"    {line}")

    csp_orders = generate_orders(
        15, oracle.zones, seed=37, start_time=8 * 60,
        start_spread=60, window_width=(30, 60),
    )
    print("\nCSP comparison (same generated 15-order instance):")
    print("  AC-3 is domain propagation; its configuration still uses backtracking.")
    csp_comparison = []
    for label, forward_checking, arc_consistency in (
        ("BT (no pruning)", False, False),
        ("BT + AC-3 (FC off)", False, True),
        ("BT + FC", True, False),
        ("BT + FC + AC-3", True, True),
    ):
        solver = AssignmentSolver(
            csp_orders, {"Van 1": 10, "Van 2": 10, "Van 3": 10},
            oracle.zones[0], 8 * 60, oracle,
        )
        result = solver.solve(
            forward_checking=forward_checking,
            arc_consistency=arc_consistency,
        )
        csp_comparison.append((label, result))
        if result.reason:
            print(f"  {label}: {result.reason}")
    print(f"  {'Configuration':<22} {'Feasible':>9} {'Assignments':>12} "
          f"{'Backtracks':>11} {'Nodes':>8} {'Values pruned':>14}")
    for label, result in csp_comparison:
        stats = result.stats
        print(
            f"  {label:<22} {str(result.feasible):>9} "
            f"{stats.assignments:>12} {stats.backtracks:>11} "
            f"{stats.nodes:>8} {stats.values_pruned:>14}"
        )
    bt_stats = csp_comparison[0][1].stats
    ac3_stats = csp_comparison[1][1].stats
    print(
        "  AC-3-only change vs BT: "
        f"backtracks {ac3_stats.backtracks - bt_stats.backtracks:+d}, "
        f"nodes {ac3_stats.nodes - bt_stats.nodes:+d}, "
        f"values pruned {ac3_stats.values_pruned - bt_stats.values_pruned:+d}."
    )

    zone_a, zone_b = oracle.zones[1:3]
    depot = oracle.zones[0]

    def conflict_time(first: int, second: int, hour: float, weekday: int) -> float:
        if first == second:
            return 0.0
        if first == depot or {first, second} != {zone_a, zone_b}:
            return 5.0
        return 20.0

    impossible = [
        Order("A", zone_a, 0, 10, demand=3),
        Order("B", zone_b, 0, 10, demand=3),
    ]
    print("\nBT vs AC-3 early failure (same two-order instance; forward checking off):")
    for label, forward_checking, arc_consistency in (
        ("BT (no pruning)", False, False),
        ("BT + AC-3", False, True),
    ):
        solver = AssignmentSolver(
            impossible, {"Van 1": 3, "Van 2": 2, "Van 3": 1},
            depot, 0, conflict_time,
        )
        result = solver.solve(forward_checking, arc_consistency)
        print(
            f"  {label}: feasible={result.feasible}, "
            f"assignments={result.stats.assignments}, "
            f"backtracks={result.stats.backtracks}, nodes={result.stats.nodes}, "
            f"values_pruned={result.stats.values_pruned}"
        )


if __name__ == "__main__":
    main()
