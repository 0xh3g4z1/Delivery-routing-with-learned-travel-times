from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from csp.assignment import (
    AssignmentSolver,
    CSPStats,
    pairwise_conflicts,
)
from integration.types import Order
from search.route import SearchResult, route_search

TravelTime = Callable[[int, int, float, int], float]


@dataclass(frozen=True)
class PlanningAttempt:
    number: int
    csp_stats: CSPStats
    route_feasible: bool | None
    blocked_pair: tuple[str, str] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PlanResult:
    assignments: dict[str, str] | None
    routes: dict[str, SearchResult]
    csp_stats: CSPStats
    reassignments: int
    reason: str | None
    window_suggestions: tuple[str, ...] = ()
    attempts: tuple[PlanningAttempt, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.assignments is not None and self.reason is None


def window_widening_suggestions(
    orders: list[Order],
    depot_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
) -> tuple[str, ...]:
    by_id = {order.order_id: order for order in orders}
    conflicts = pairwise_conflicts(
        orders, depot_zone, start_time, travel_time, weekday
    )
    suggestions = []
    for first_id, second_id in sorted(conflicts):
        first = by_id[first_id]
        second = by_id[second_id]
        options = []
        for leading, following in ((first, second), (second, first)):
            first_travel = max(0.0, float(travel_time(
                depot_zone, leading.zone, (start_time // 60) % 24, weekday
            )))
            first_arrival = start_time + first_travel
            if first_arrival > leading.window_end:
                options.append((first_arrival - leading.window_end, leading, first_arrival))
                continue
            first_service_start = max(first_arrival, leading.window_start)
            after_first = first_service_start + leading.service_minutes
            second_travel = max(0.0, float(travel_time(
                leading.zone, following.zone,
                (after_first // 60) % 24, weekday,
            )))
            second_arrival = after_first + second_travel
            if second_arrival > following.window_end:
                options.append((
                    second_arrival - following.window_end,
                    following,
                    second_arrival,
                ))
        if not options:
            continue
        _, order, needed_arrival = min(options, key=lambda value: value[0])
        steps = max(1, int((needed_arrival - order.window_end + 14) // 15))
        widened_end = order.window_end + 15 * steps
        if widened_end > 24 * 60:
            suggestions.append(
                f"{order.order_id}: required extension exceeds 24:00; "
                "widening within the same day is not enough."
            )
        else:
            suggestions.append(
                f"{order.order_id}: consider extending its latest time by "
                f"{steps} x 15 minutes to "
                f"{widened_end // 60:02d}:{widened_end % 60:02d}."
            )
    if not suggestions:
        suggestions.append(
            "No pair-specific time-window widening was identified; if capacity "
            "is the cause, add capacity or another van instead."
        )
    return tuple(suggestions)


def generate_orders(
    count: int,
    zones: list[int],
    seed: int = 42,
    start_time: int = 8 * 60,
    start_spread: int = 120,
    window_width: tuple[int, int] = (300, 420),
) -> list[Order]:
    if count < 0:
        raise ValueError("count cannot be negative")
    if count and not zones:
        raise ValueError("At least one real zone is required to generate orders")
    if start_spread < 0 or window_width[0] < 1 or window_width[1] < window_width[0]:
        raise ValueError("Invalid generated time-window settings")
    rng = random.Random(seed)
    orders = []
    for index in range(count):
        earliest = start_time + rng.randint(0, start_spread)
        duration = rng.randint(*window_width)
        orders.append(Order(
            order_id=f"O{index + 1:02d}",
            zone=rng.choice(zones),
            window_start=earliest,
            window_end=earliest + duration,
            demand=rng.randint(1, 2),
            service_minutes=float(rng.randint(5, 10)),
            synthetic=True,
        ))
    return orders


def plan_orders(
    orders: list[Order],
    van_capacities: dict[str, int],
    depot_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
    max_reassignments: int = 5,
    max_search_expansions: int = 250_000,
) -> PlanResult:
    extra_conflicts: set[tuple[str, str]] = set()
    last_stats = CSPStats(0, 0, 0)
    attempts: list[PlanningAttempt] = []
    for attempt in range(max_reassignments + 1):
        solver = AssignmentSolver(
            orders, van_capacities, depot_zone, start_time,
            travel_time, weekday, extra_conflicts,
        )
        assignment_result = solver.solve(
            forward_checking=True, arc_consistency=True
        )
        last_stats = assignment_result.stats
        if not assignment_result.feasible:
            attempts.append(PlanningAttempt(
                attempt + 1, last_stats, None, reason=assignment_result.reason,
            ))
            return PlanResult(
                None, {}, last_stats, attempt, assignment_result.reason,
                window_widening_suggestions(
                    orders, depot_zone, start_time, travel_time, weekday
                ) if assignment_result.complete else (), tuple(attempts),
            )
        assignments = assignment_result.assignments
        routes: dict[str, SearchResult] = {}
        blocked_pair = None
        for van_id in van_capacities:
            van_orders = [
                order for order in orders
                if assignments[order.order_id] == van_id
            ]
            result = route_search(
                van_orders, depot_zone, start_time, travel_time,
                algorithm="astar", weekday=weekday,
                max_expansions=max_search_expansions,
            )
            routes[van_id] = result
            if not result.feasible:
                if not result.complete:
                    attempts.append(PlanningAttempt(
                        attempt + 1, last_stats, False,
                        blocked_pair=result.blocked_pair,
                        reason=result.reason,
                    ))
                    return PlanResult(
                        assignments, routes, last_stats, attempt,
                        result.reason,
                        window_widening_suggestions(
                            orders, depot_zone, start_time, travel_time, weekday
                        ), tuple(attempts),
                    )
                if result.blocked_pair is not None:
                    blocked_pair = tuple(sorted(result.blocked_pair))
                break
        if len(routes) == len(van_capacities) and all(
            route.feasible for route in routes.values()
        ):
            attempts.append(PlanningAttempt(attempt + 1, last_stats, True))
            return PlanResult(
                assignments, routes, last_stats, attempt, None,
                attempts=tuple(attempts),
            )
        attempts.append(PlanningAttempt(
            attempt + 1, last_stats, False, blocked_pair=blocked_pair,
            reason=next(
                (route.reason for route in routes.values() if not route.feasible),
                None,
            ),
        ))
        if blocked_pair is None:
            return PlanResult(
                assignments, routes, last_stats, attempt,
                "No feasible route found and no additional pairwise conflict "
                "could be identified.",
                window_widening_suggestions(
                    orders, depot_zone, start_time, travel_time, weekday
                ), tuple(attempts),
            )
        if blocked_pair in extra_conflicts:
            return PlanResult(
                assignments, routes, last_stats, attempt,
                f"Route remained infeasible after adding conflict {blocked_pair}.",
                window_widening_suggestions(
                    orders, depot_zone, start_time, travel_time, weekday
                ), tuple(attempts),
            )
        if attempt == max_reassignments:
            return PlanResult(
                assignments, routes, last_stats, attempt,
                f"No feasible plan after {max_reassignments} reassignment loops.",
                window_widening_suggestions(
                    orders, depot_zone, start_time, travel_time, weekday
                ), tuple(attempts),
            )
        extra_conflicts.add(blocked_pair)
    return PlanResult(
        None, {}, last_stats, max_reassignments, "Planning failed.",
        attempts=tuple(attempts),
    )
