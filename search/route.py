from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Callable, Literal

from integration.models import Order

Algorithm = Literal["astar", "ucs", "greedy"]
TravelTime = Callable[[int, int, float, int], float]


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    route: tuple[Order, ...] | None
    arrivals: dict[str, float]
    cost: float | None
    expanded: int
    generated: int
    complete: bool
    reason: str | None = None
    blocked_pair: tuple[str, str] | None = None
    trace: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.route is not None and self.complete


def _heuristic(
    mask: int,
    orders: list[Order],
    min_incoming: list[float],
) -> float:
    return sum(
        min_incoming[index]
        for index in range(len(orders))
        if not mask & (1 << index)
    )


def _minimum_incoming_costs(
    orders: list[Order],
    start_zone: int,
    travel_time: TravelTime,
    weekday: int,
) -> list[float]:
    result: list[float] = []
    for target_index, target in enumerate(orders):
        source_zones = {start_zone}
        source_zones.update(
            order.zone for index, order in enumerate(orders)
            if index != target_index
        )
        best = float("inf")
        for source in source_zones:
            if source == target.zone:
                best = 0.0
                break
            for hour in range(24):
                best = min(
                    best,
                    max(0.0, float(travel_time(source, target.zone, hour, weekday))),
                )
        result.append(0.0 if best == float("inf") else best)
    return result


def route_search(
    orders: list[Order],
    start_zone: int,
    start_time: int,
    travel_time: TravelTime,
    algorithm: Algorithm = "astar",
    weekday: int = 0,
    max_expansions: int = 250_000,
) -> SearchResult:
    if algorithm not in {"astar", "ucs", "greedy"}:
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    if max_expansions < 1:
        raise ValueError("max_expansions must be positive")
    orders = list(orders)
    if not orders:
        return SearchResult(algorithm, (), {}, 0.0, 0, 1, True)

    minimum_incoming = _minimum_incoming_costs(
        orders, start_zone, travel_time, weekday
    )
    counter = itertools.count()
    initial_h = _heuristic(0, orders, minimum_incoming)
    initial_priority = initial_h if algorithm != "ucs" else 0.0
    queue = [(initial_priority, next(counter), 0.0, 0, -1,
              float(start_time), ())]
    best_cost: dict[tuple[int, int], float] = {(0, -1): 0.0}
    expanded = 0
    generated = 1
    blocked_pair: tuple[str, str] | None = None
    trace: list[str] = []
    full_mask = (1 << len(orders)) - 1

    while queue and expanded < max_expansions:
        _, _, cost, mask, last_index, clock, path = heapq.heappop(queue)
        state = (mask, last_index)
        if cost > best_cost.get(state, float("inf")) + 1e-9:
            continue
        if mask == full_mask:
            route = tuple(orders[index] for index in path)
            arrivals = _replay_arrivals(
                route, start_zone, start_time, travel_time, weekday
            )
            return SearchResult(
                algorithm, route, arrivals, cost, expanded, generated, True,
                trace=tuple(trace),
            )
        expanded += 1
        current_zone = start_zone if last_index < 0 else orders[last_index].zone
        if len(trace) < 12:
            visited = ",".join(orders[index].order_id for index in path) or "-"
            remaining_estimate = _heuristic(
                mask, orders, minimum_incoming
            )
            trace.append(
                f"expand {expanded}: zone={current_zone}, visited={visited}, "
                f"clock={clock:.1f}, g={cost:.1f}, h={remaining_estimate:.1f}"
            )

        for index, order in enumerate(orders):
            bit = 1 << index
            if mask & bit:
                continue
            travel = max(0.0, float(travel_time(
                current_zone, order.zone, (clock // 60) % 24, weekday
            )))
            arrival = clock + travel
            if arrival > order.window_end:
                if last_index >= 0 and blocked_pair is None:
                    blocked_pair = (
                        orders[last_index].order_id, order.order_id
                    )
                continue
            service_start = max(arrival, float(order.window_start))
            next_clock = service_start + order.service_minutes
            next_cost = cost + travel + (service_start - arrival) + order.service_minutes
            next_mask = mask | bit
            next_state = (next_mask, index)
            if next_cost >= best_cost.get(next_state, float("inf")) - 1e-9:
                continue
            best_cost[next_state] = next_cost
            next_path = path + (index,)
            h = _heuristic(next_mask, orders, minimum_incoming)
            if algorithm == "astar":
                priority = next_cost + h
            elif algorithm == "greedy":
                priority = h
            else:
                priority = next_cost
            heapq.heappush(
                queue,
                (priority, next(counter), next_cost, next_mask, index,
                 next_clock, next_path),
            )
            generated += 1

    if queue:
        reason = f"Search limit reached after {max_expansions} expanded states."
        complete = False
    else:
        reason = "No feasible route visits every assigned order within its time window."
        complete = True
    return SearchResult(
        algorithm, None, {}, None, expanded, generated, complete, reason,
        blocked_pair, tuple(trace),
    )


def _replay_arrivals(
    route: tuple[Order, ...],
    start_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    zone = start_zone
    clock = float(start_time)
    for order in route:
        duration = max(0.0, float(travel_time(
            zone, order.zone, (clock // 60) % 24, weekday
        )))
        arrival = clock + duration
        result[order.order_id] = arrival
        clock = max(arrival, float(order.window_start)) + order.service_minutes
        zone = order.zone
    return result


def brute_force_route(
    orders: list[Order],
    start_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
) -> tuple[float, tuple[Order, ...]] | None:
    best: tuple[float, tuple[Order, ...]] | None = None
    for permutation in itertools.permutations(orders):
        zone = start_zone
        clock = float(start_time)
        cost = 0.0
        feasible = True
        for order in permutation:
            duration = max(0.0, float(travel_time(
                zone, order.zone, (clock // 60) % 24, weekday
            )))
            arrival = clock + duration
            if arrival > order.window_end:
                feasible = False
                break
            service_start = max(arrival, float(order.window_start))
            cost += duration + service_start - arrival + order.service_minutes
            clock = service_start + order.service_minutes
            zone = order.zone
        if feasible and (best is None or cost < best[0]):
            best = (cost, tuple(permutation))
    return best
