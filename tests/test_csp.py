from csp.assignment import AssignmentSolver
from integration.models import Order


def _short_travel(first: int, second: int, hour: float, weekday: int) -> float:
    if first == second:
        return 0.0
    if first == 1:
        return 5.0
    if {first, second} == {2, 3}:
        return 20.0
    return 5.0


def test_assigns_orders_with_capacity():
    orders = [
        Order("A", 2, 0, 100, demand=1),
        Order("B", 3, 0, 100, demand=1),
        Order("C", 4, 0, 100, demand=1),
    ]
    result = AssignmentSolver(
        orders, {"V1": 2, "V2": 2}, 1, 0, _short_travel
    ).solve()
    assert result.feasible
    assert set(result.assignments) == {"A", "B", "C"}


def test_forward_checking_keeps_conflicting_orders_apart():
    orders = [
        Order("A", 2, 0, 10),
        Order("B", 3, 0, 10),
    ]
    solver = AssignmentSolver(orders, {"V1": 2, "V2": 2}, 1, 0, _short_travel)
    result = solver.solve(forward_checking=True, arc_consistency=False)
    assert result.feasible
    assert result.assignments["A"] != result.assignments["B"]


def test_ac3_can_be_enabled_without_forward_checking():
    orders = [
        Order("A", 2, 0, 10),
        Order("B", 3, 0, 10),
    ]
    solver = AssignmentSolver(orders, {"V1": 2, "V2": 2}, 1, 0, _short_travel)
    result = solver.solve(forward_checking=False, arc_consistency=True)
    assert result.feasible
    assert result.assignments["A"] != result.assignments["B"]


def test_solver_counters_are_populated():
    orders = [
        Order("A", 2, 0, 100),
        Order("B", 3, 0, 100),
    ]
    result = AssignmentSolver(
        orders, {"V1": 2, "V2": 2}, 1, 0, _short_travel
    ).solve()
    assert result.stats.nodes > 0
    assert result.stats.assignments > 0


def test_seeded_order_generation_is_reproducible():
    from integration.planner import generate_orders

    first = generate_orders(5, [1, 2, 3], seed=7)
    second = generate_orders(5, [1, 2, 3], seed=7)
    assert first == second
    assert all(order.synthetic for order in first)


def test_empty_csp_has_empty_solution():
    result = AssignmentSolver([], {"V1": 5}, 1, 0, _short_travel).solve()
    assert result.feasible
    assert result.assignments == {}


def test_unreachable_window_is_an_impossible_case():
    order = Order("late", 2, 0, 1)
    result = AssignmentSolver(
        [order], {"V1": 5}, 1, 0, _short_travel
    ).solve()
    assert not result.feasible
    assert result.reason is not None
    assert "no solution" in result.reason.lower()
    assert "cannot be served from the depot" in result.reason


def test_demand_above_every_capacity_is_an_impossible_case():
    order = Order("heavy", 2, 0, 100, demand=3)
    result = AssignmentSolver(
        [order], {"V1": 2, "V2": 2}, 1, 0, _short_travel
    ).solve()
    assert not result.feasible
    assert result.reason is not None
    assert "no solution" in result.reason.lower()
    assert "requires 3 capacity units" in result.reason
    assert "largest van capacity is 2" in result.reason


def test_backtracking_detects_single_van_pair_conflict():
    orders = [
        Order("A", 2, 0, 10),
        Order("B", 3, 0, 10),
    ]
    result = AssignmentSolver(
        orders, {"only van": 2}, 1, 0, _short_travel
    ).solve()
    assert not result.feasible
    assert "no solution" in result.reason.lower()


def test_impossible_pair_gets_a_15_minute_widening_suggestion():
    from integration.planner import window_widening_suggestions

    orders = [
        Order("A", 2, 0, 10),
        Order("B", 3, 0, 10),
    ]
    suggestions = window_widening_suggestions(
        orders, 1, 0, _short_travel
    )
    assert suggestions
    assert any("15 minutes" in suggestion for suggestion in suggestions)


def test_search_limit_is_not_reported_as_proven_impossible():
    orders = [
        Order("A", 2, 0, 100),
        Order("B", 3, 0, 100),
    ]
    result = AssignmentSolver(
        orders, {"V1": 2, "V2": 2}, 1, 0, _short_travel
    ).solve(max_nodes=1)
    assert not result.complete
    assert "unknown" in result.reason.lower()
