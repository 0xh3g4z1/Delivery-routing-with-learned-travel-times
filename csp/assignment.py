from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Iterable

from integration.types import Order

TravelTime = Callable[[int, int, float, int], float]
Pair = tuple[str, str]


@dataclass(frozen=True)
class CSPStats:
    assignments: int
    backtracks: int
    nodes: int
    values_pruned: int = 0


@dataclass(frozen=True)
class CSPResult:
    assignments: dict[str, str] | None
    stats: CSPStats
    reason: str | None = None
    complete: bool = True

    @property
    def feasible(self) -> bool:
        return self.assignments is not None


def _arrival(
    from_zone: int,
    order: Order,
    clock: float,
    travel_time: TravelTime,
    weekday: int,
) -> tuple[float, float]:
    duration = max(0.0, float(travel_time(
        from_zone, order.zone, (clock // 60) % 24, weekday
    )))
    arrival = clock + duration
    start_service = max(arrival, float(order.window_start))
    return arrival, start_service + order.service_minutes


def can_deliver_from_depot(
    order: Order,
    depot_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
) -> bool:
    arrival, _ = _arrival(depot_zone, order, start_time, travel_time, weekday)
    return arrival <= order.window_end


def pair_can_share_van(
    first: Order,
    second: Order,
    depot_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
) -> bool:
    for leading, following in ((first, second), (second, first)):
        if not can_deliver_from_depot(
            leading, depot_zone, start_time, travel_time, weekday
        ):
            continue
        _, after_leading = _arrival(
            depot_zone, leading, start_time, travel_time, weekday
        )
        arrival, _ = _arrival(
            leading.zone, following, after_leading, travel_time, weekday
        )
        if arrival <= following.window_end:
            return True
    return False


def pairwise_conflicts(
    orders: Iterable[Order],
    depot_zone: int,
    start_time: int,
    travel_time: TravelTime,
    weekday: int = 0,
) -> set[Pair]:
    items = list(orders)
    conflicts: set[Pair] = set()
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            if not pair_can_share_van(
                first, second, depot_zone, start_time, travel_time, weekday
            ):
                conflicts.add(tuple(sorted((first.order_id, second.order_id))))
    return conflicts


class AssignmentSolver:
    def __init__(
        self,
        orders: list[Order],
        van_capacities: dict[str, int],
        depot_zone: int,
        start_time: int,
        travel_time: TravelTime,
        weekday: int = 0,
        extra_conflicts: Iterable[Pair] = (),
    ) -> None:
        self.orders = list(orders)
        self.capacities = dict(van_capacities)
        self.depot_zone = depot_zone
        self.start_time = start_time
        self.travel_time = travel_time
        self.weekday = weekday
        self.conflicts = pairwise_conflicts(
            self.orders, depot_zone, start_time, travel_time, weekday
        )
        self.conflicts.update(tuple(sorted(pair)) for pair in extra_conflicts)
        self.by_id = {order.order_id: order for order in self.orders}
        self.neighbors: dict[str, set[str]] = defaultdict(set)
        for first, second in self.conflicts:
            self.neighbors[first].add(second)
            self.neighbors[second].add(first)
        self.assignment_count = 0
        self.backtrack_count = 0
        self.node_count = 0
        self.pruned_count = 0
        self.max_nodes = 1_000_000
        self.limit_reached = False

    def solve(
        self,
        forward_checking: bool = True,
        arc_consistency: bool = True,
        max_nodes: int = 1_000_000,
    ) -> CSPResult:
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.assignment_count = 0
        self.backtrack_count = 0
        self.node_count = 0
        self.pruned_count = 0
        self.limit_reached = False
        self.max_nodes = max_nodes
        domains: dict[str, set[str]] = {}
        for order in self.orders:
            reachable = can_deliver_from_depot(
                order, self.depot_zone, self.start_time,
                self.travel_time, self.weekday,
            )
            domains[order.order_id] = {
                van_id for van_id, capacity in self.capacities.items()
                if capacity >= order.demand and reachable
            }
            if not domains[order.order_id]:
                reason = (
                    f"No solution: order {order.order_id} cannot be served "
                    "from the depot in its time window or exceeds every van capacity."
                )
                return self._result(None, reason)

        if arc_consistency and not self._ac3(domains):
            return self._result(None, "No solution: AC-3 emptied an order's van domain.")
        solution = self._backtrack(
            {}, domains, forward_checking, arc_consistency
        )
        if self.limit_reached:
            return self._result(
                None,
                f"CSP search limit reached after {max_nodes} nodes; "
                "feasibility is unknown.",
                complete=False,
            )
        if solution is None:
            return self._result(None, "No solution: van, capacity, and time-window constraints conflict.")
        return self._result(solution, None)

    def _result(
        self,
        solution: dict[str, str] | None,
        reason: str | None,
        complete: bool = True,
    ) -> CSPResult:
        return CSPResult(
            assignments=solution,
            stats=CSPStats(
                assignments=self.assignment_count,
                backtracks=self.backtrack_count,
                nodes=self.node_count,
                values_pruned=self.pruned_count,
            ),
            reason=reason,
            complete=complete,
        )

    def _consistent_with_assignment(
        self,
        order_id: str,
        van_id: str,
        assignment: dict[str, str],
    ) -> bool:
        if any(assignment.get(other) == van_id for other in self.neighbors[order_id]):
            return False
        load = sum(
            self.by_id[other].demand
            for other, assigned_van in assignment.items()
            if assigned_van == van_id
        )
        return load + self.by_id[order_id].demand <= self.capacities[van_id]

    def _backtrack(
        self,
        assignment: dict[str, str],
        domains: dict[str, set[str]],
        forward_checking: bool,
        arc_consistency: bool,
    ) -> dict[str, str] | None:
        if self.node_count >= self.max_nodes:
            self.limit_reached = True
            return None
        self.node_count += 1
        if len(assignment) == len(self.orders):
            return assignment
        unassigned = [order.order_id for order in self.orders
                      if order.order_id not in assignment]
        if forward_checking:
            variable = min(unassigned, key=lambda key: (len(domains[key]), key))
        else:
            variable = unassigned[0]

        attempted = False
        for van_id in sorted(domains[variable]):
            if not self._consistent_with_assignment(variable, van_id, assignment):
                continue
            attempted = True
            self.assignment_count += 1
            next_assignment = dict(assignment)
            next_assignment[variable] = van_id
            next_domains = {key: set(values) for key, values in domains.items()}
            next_domains[variable] = {van_id}

            if forward_checking and not self._forward_check(
                variable, van_id, next_assignment, next_domains
            ):
                self.backtrack_count += 1
                continue
            if arc_consistency and not self._ac3(next_domains):
                self.backtrack_count += 1
                continue
            result = self._backtrack(
                next_assignment, next_domains, forward_checking, arc_consistency
            )
            if result is not None:
                return result
            if self.limit_reached:
                return None
            self.backtrack_count += 1
        if not attempted:
            self.backtrack_count += 1
        return None

    def _forward_check(
        self,
        selected: str,
        van_id: str,
        assignment: dict[str, str],
        domains: dict[str, set[str]],
    ) -> bool:
        for neighbor in self.neighbors[selected]:
            if neighbor not in assignment:
                before = len(domains[neighbor])
                domains[neighbor].discard(van_id)
                self.pruned_count += before - len(domains[neighbor])
                if not domains[neighbor]:
                    return False
        loads: dict[str, int] = defaultdict(int)
        for order_id, assigned_van in assignment.items():
            loads[assigned_van] += self.by_id[order_id].demand
        for order in self.orders:
            if order.order_id in assignment:
                continue
            before = len(domains[order.order_id])
            domains[order.order_id] = {
                van for van in domains[order.order_id]
                if loads[van] + order.demand <= self.capacities[van]
            }
            self.pruned_count += before - len(domains[order.order_id])
            if not domains[order.order_id]:
                return False
        return True

    def _ac3(self, domains: dict[str, set[str]]) -> bool:
        arcs = []
        for first, second in self.conflicts:
            arcs.extend(((first, second), (second, first)))
        queue = deque(arcs)
        while queue:
            first, second = queue.popleft()
            revised = {
                value for value in domains[first]
                if any(value != other for other in domains[second])
            }
            if revised == domains[first]:
                continue
            self.pruned_count += len(domains[first]) - len(revised)
            domains[first] = revised
            if not revised:
                return False
            for neighbor in self.neighbors[first] - {second}:
                queue.append((neighbor, first))
        return True
