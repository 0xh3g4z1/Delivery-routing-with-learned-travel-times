from pathlib import Path
import subprocess
import sys

from integration.models import Order
from integration.planner import plan_orders
from run_demo import _compare_route_search, _generate_scenario_orders


def _travel(first: int, second: int, hour: float, weekday: int) -> float:
    return 0.0 if first == second else float(abs(first - second) + 1)


def test_training_script_supports_direct_file_execution():
    root = Path(__file__).resolve().parents[1]
    script = root / "nn" / "train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-iter" in result.stdout


def test_demo_compares_route_search_methods_on_the_same_assignments():
    orders = [
        Order("A", 2, 0, 500),
        Order("B", 5, 0, 500),
        Order("C", 3, 0, 500),
    ]
    results = _compare_route_search(
        orders,
        {"A": "Van 1", "B": "Van 1", "C": "Van 2"},
        {"Van 1": 5, "Van 2": 5},
        depot_zone=1,
        start_time=0,
        travel_time=_travel,
        weekday=0,
    )

    assert [result["algorithm"] for result in results] == ["A*", "UCS", "Greedy"]
    assert all(result["feasible"] for result in results)
    assert results[0]["route_cost_minutes"] == results[1]["route_cost_minutes"]
    assert all(result["nodes_expanded"] > 0 for result in results)


def test_impossible_demo_scenario_reports_no_solution():
    capacities = {"Van 1": 10, "Van 2": 10, "Van 3": 10}
    orders = _generate_scenario_orders("impossible", [42, 43], 42, capacities)

    result = plan_orders(
        orders, capacities, depot_zone=42, start_time=8 * 60,
        travel_time=_travel, weekday=0,
    )

    assert len(orders) == 1
    assert orders[0].demand == 11
    assert not result.feasible
    assert result.reason is not None
    assert "no solution" in result.reason.lower()


def test_outside_window_scenario_cannot_be_served_on_time():
    capacities = {"Van 1": 10, "Van 2": 10, "Van 3": 10}
    orders = _generate_scenario_orders(
        "outside-window", [42, 43], 42, capacities
    )

    result = plan_orders(
        orders, capacities, depot_zone=42, start_time=8 * 60,
        travel_time=_travel, weekday=0,
    )

    assert orders[0].demand <= min(capacities.values())
    assert orders[0].window_end == 8 * 60 - 1
    assert not result.feasible
    assert result.reason is not None
    assert "no solution" in result.reason.lower()
    assert "within its time window" in result.reason
    assert any("extend" in item.lower() for item in result.window_suggestions)
