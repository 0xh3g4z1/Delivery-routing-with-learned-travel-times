from integration.types import Order
from search.route import brute_force_route, route_search


def _travel(first: int, second: int, hour: float, weekday: int) -> float:
    return 0.0 if first == second else float(abs(first - second) + 1)


def test_astar_matches_brute_force_for_six_stops():
    orders = [
        Order(f"O{index}", zone=index + 1, window_start=0, window_end=600)
        for index in range(2, 8)
    ]
    expected = brute_force_route(orders, 1, 0, _travel)
    actual = route_search(orders, 1, 0, _travel, algorithm="astar")
    assert expected is not None
    assert actual.feasible
    assert actual.cost == expected[0]
    assert len(actual.route) == 6


def test_uniform_cost_matches_brute_force():
    orders = [
        Order("A", 2, 0, 100),
        Order("B", 3, 0, 100),
        Order("C", 4, 0, 100),
    ]
    expected = brute_force_route(orders, 1, 0, _travel)
    actual = route_search(orders, 1, 0, _travel, algorithm="ucs")
    assert actual.feasible
    assert actual.cost == expected[0]


def test_greedy_finds_a_feasible_open_window_route():
    orders = [
        Order("A", 4, 0, 100),
        Order("B", 2, 0, 100),
        Order("C", 3, 0, 100),
    ]
    result = route_search(orders, 1, 0, _travel, algorithm="greedy")
    assert result.feasible
    assert {order.order_id for order in result.route} == {"A", "B", "C"}


def test_route_arrivals_obey_windows():
    orders = [
        Order("A", 2, 10, 30),
        Order("B", 3, 20, 40),
    ]
    result = route_search(orders, 1, 0, _travel, algorithm="astar")
    assert result.feasible
    assert all(result.arrivals[order.order_id] <= order.window_end for order in orders)


def test_six_stop_brute_force_reports_known_route():
    orders = [
        Order(f"Z{zone}", zone, 0, 500)
        for zone in (2, 3, 4, 5, 6, 7)
    ]
    result = brute_force_route(orders, 1, 0, _travel)
    assert result is not None
    assert len(result[1]) == 6


def test_empty_route_is_a_zero_cost_solution():
    result = route_search([], 1, 480, _travel)
    assert result.feasible
    assert result.route == ()
    assert result.cost == 0


def test_single_order_route():
    result = route_search([Order("only", 2, 0, 20)], 1, 0, _travel)
    assert result.feasible
    assert result.route[0].order_id == "only"


def test_search_limit_is_reported_as_incomplete():
    orders = [
        Order(f"O{index}", index + 2, 0, 1000)
        for index in range(6)
    ]
    result = route_search(
        orders, 1, 0, _travel, algorithm="ucs", max_expansions=1
    )
    assert not result.complete
    assert "limit" in result.reason.lower()


def test_unreachable_order_returns_no_solution():
    result = route_search(
        [Order("late", 9, 0, 1)], 1, 0, _travel, algorithm="astar"
    )
    assert not result.feasible
    assert result.reason is not None
    assert "no feasible route" in result.reason.lower()
