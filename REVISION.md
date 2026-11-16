# Revision guide: learned-time delivery routing

This file is the study guide for the code in this workspace. It explains the
technical terms, the data flow, and the reasoning behind each algorithm in
plain language.

## 1. The project in one picture

**Real September TLC trips** → clean and split by date → train a neural network
and baselines → predict zone-to-zone travel minutes → create an explicitly
synthetic order instance using real zone IDs → CSP assigns orders to vans →
A* chooses each van's visit order → compare methods and save tables/plots.

The first half is based on real observations. The planning inputs are
synthetic because the TLC source does not publish delivery windows, van
capacities, or courier orders. Keeping those two kinds of information separate
is essential when explaining the results.

## 2. Data and machine-learning terms

### Parquet

Parquet is a compressed, column-oriented file format. Reading only the
necessary columns in batches saves memory compared with loading every fare,
flag, and identifier into one large table.

### TLC zone

A TLC `LocationID` identifies a geographic taxi zone. `taxi_zone_lookup.csv`
maps the ID to a borough and human-readable zone. A zone is an area, not an
exact road coordinate or distance.

### Feature and target

- A **feature** is an input used to make a prediction.
- The **target** is the value the model learns to predict.
- Features here: pickup zone, drop-off zone, hour, weekday, and training-only
  mean trip distance for the zone pair.
- Target here: trip duration in minutes.

For Yellow and Green trips, duration is drop-off timestamp minus pickup
timestamp. FHVHV supplies a `trip_time` value in seconds, divided by 60.
Records with durations outside 1–120 minutes are excluded.

### Training, test set, leakage

The model sees September 1–23 (weekdays only) during training. It is evaluated
on weekdays from September 24–30. This is a time-based split: later days are
held out rather than randomly mixed into training.

**Data leakage** is when information from the evaluation period affects the
model's training. To avoid it, the 20 busiest zones and zone-pair mean
distances are computed from training records only. An unseen pair uses the
global training mean distance.

### MLP / neural network

`MLPRegressor` is a multilayer perceptron: input values pass through
intermediate layers of learned weights and nonlinear activations, then produce
a duration estimate. This code uses two hidden layers with 32 and 16 units.
Zone/hour/day categories are one-hot encoded; mean distance is standardized.
The optimizer is Adam. The fixed random seed makes initialization and
sampling reproducible, though different library/hardware versions can still
produce small numeric differences.

### Baseline

A **baseline** is a simpler reference method. A complex model is only useful
if it beats a reasonable baseline.

- **Historical pair average:** mean observed training duration for the same
  pickup/drop-off zones; unseen pairs fall back to the training global mean.
- **Distance only:** training median minutes-per-mile multiplied by the
  training-only mean distance for the zone pair.

### MAE and RMSE

- **MAE** (mean absolute error) is the average absolute prediction error in
  minutes. It is easy to interpret and is less sensitive to a few large errors.
- **RMSE** (root mean squared error) squares errors before averaging, so large
  misses count more heavily. It is also measured in minutes.

Lower values are better. Compare all models on the same held-out trips.

## 3. Planning terms

### Order, time window, service time, demand

An order is a generated planning task with a destination zone, earliest and
latest allowed service-start times, a capacity-unit demand, and a service
duration. These values are **scenario data**, not TLC observations.

The time values are minutes after midnight. A van may arrive before the window
opens and wait; arriving after the window closes makes that order infeasible.
The service duration advances the van's clock after arrival.

### Vehicle capacity

Capacity is a per-van limit on the sum of assigned order demands. This is a
simple scenario-level capacity model, not an observed attribute of any taxi in
the TLC files.

### Variable, domain, and constraint

In the CSP:

- A **variable** is one order.
- Its **domain** is the set of vans that could serve it.
- A **constraint** removes assignments that cannot work.

Unary constraints check whether an order is reachable from the depot within
its window and whether at least one van has enough capacity. A binary
`different van` constraint is added for a pair of orders if neither visit
sequence can serve both within their windows.

### Backtracking (BT)

Backtracking tries a value for one order, continues recursively, and returns
to the previous choice if a constraint becomes impossible. It is complete:
given enough time, it finds a solution if one exists or reports no solution.
The counter `assignments` is the number of attempted variable/value choices;
`backtracks` counts failed branches; `nodes` counts recursive search states.
`values_pruned` counts van choices removed by forward checking or AC-3.

### Forward checking (FC)

After assigning an order to a van, forward checking removes that van from
conflicting orders' domains and removes vans that no longer have enough
capacity. If any remaining order has an empty domain, it backtracks
immediately instead of exploring deeper.

### Arc consistency and AC-3

Two domains joined by a `different van` constraint are arc-consistent if every
van value on one side has at least one *different* supporting van value on the
other side. **AC-3** repeatedly checks such directed arcs and removes
unsupported values. If a domain becomes empty, the current branch is
impossible. The implementation allows AC-3 to be toggled independently from
forward checking; the required comparison also uses BT, BT+FC, and
BT+FC+AC-3. `python -m eval.trace_demo` separately compares plain BT with
BT+AC-3 while forward checking is disabled, so the effect of AC-3 can be read
without conflating it with forward checking. AC-3 is propagation performed
within the backtracking solver, not a separate search algorithm.

### Search state and action

For one van, the route-search state consists of:

1. the last visited location (or starting depot),
2. a bit mask recording which assigned orders are complete,
3. the current clock time.

An **action** is to visit one uncompleted assigned order. The predicted trip
time depends on the zone pair, hour, and weekday. The cost adds travel,
waiting for a window to open, and scenario service time.

### Uniform-cost search (UCS)

UCS expands the state with the lowest accumulated route cost first. It returns
an optimal route if allowed to finish, but may expand many states.

### Greedy best-first search

Greedy prioritizes the estimated remaining cost and does not include the cost
already spent when ranking states. It can find a route quickly, but is not
guaranteed to find the cheapest route.

### A* and admissible heuristic

A* ranks states by `g + h`:

- `g` is the real cost so far (travel + waiting + service).
- `h` estimates the remaining cost.

The heuristic sums, for every unvisited order, its cheapest possible incoming
travel time from the start or another order, across the day. Every actual
route must enter every remaining order once, and each selected incoming edge
cannot be cheaper than that order's minimum. Therefore the sum cannot
overstate the remaining travel cost; it is **admissible**. An admissible
heuristic lets A* preserve optimality while often exploring fewer states.

The six-stop test compares A* with exhaustive permutation enumeration, an
independent exact answer for a small instance.

### Expanded/generated nodes

- **Expanded**: a state removed from the priority queue and whose next actions
  are examined.
- **Generated**: a successor state added to the priority queue.

These are algorithm work counters, not counts of real taxi trips.

## 4. How the parts cooperate

1. The trained model creates travel-time estimates for zone pairs by hour.
2. The CSP assigns generated orders to vans, respecting capacity, depot
   reachability, and pairwise time-window incompatibilities.
3. A* finds a feasible low-cost stop order for each van.
4. If a route proves that a pair cannot share that van, the integration adds a
   different-van constraint and runs the CSP again, for at most five
   reassignment loops.
5. If assignment or route search fails, the program prints a clear reason
   rather than pretending that a plan exists.

For time-window failures, the program offers candidate end-time extensions in
15-minute steps. These are suggestions only; it does not change order data
without an explicit dispatch decision. Capacity failures call for capacity or
fleet changes instead of wider windows.

## 5. Experiments

1. **Search:** 6–10 stops, A*/UCS/Greedy, five fixed seeds. Report route cost,
   expanded/generated nodes, run time, and whether a search reached its limit.
2. **CSP:** 15/20/30 orders, BT/BT+FC/BT+FC+AC-3, five fixed seeds. Report
   assignments tried, backtracks, nodes, constraint count, feasibility, and
   run time.
3. **Held-out replay:** five seeded samples of held-out real trips compare the
   three travel-time methods. Because no actual delivery deadline exists, a
   clearly labeled synthetic allowance of five minutes is added to each
   observed trip duration. A prediction beyond that budget is counted as a
   simulated-late proxy, not a real late delivery.

All summaries include mean and sample standard deviation across the five
seeds. CSV files retain per-seed observations so the tables can be checked.

## 6. Important limitations to say out loud

- Taxi trips are evidence about taxi trip times, not courier delivery jobs.
- A zone-to-zone travel estimate loses road-level detail and the pickup point
  inside each zone.
- The generated 3-van capacity/time-window problem demonstrates algorithms; it
  does not measure a real dispatcher or fleet.
- The replay has no real delivery deadline or delivery outcome. Its late count
  is only a comparison proxy with an explicitly generated allowance.
- TLC says trip records can contain inaccuracies.
- A reproducible seed controls code-generated randomness, but is not proof that
  a generated scenario represents a real operation.

## 7. Files to revise

- `data/make_data.py`: source ingestion, filtering, train-only statistics,
  seeded sampling.
- `nn/model.py` and `nn/train.py`: model, baselines, metrics, artifact.
- `csp/assignment.py`: different-van constraints, BT, FC, AC-3, counters.
- `search/route.py`: A*, UCS, Greedy, brute-force checker, counters.
- `integration/models.py`: order data model used by the CSP and route search.
- `integration/planner.py`: generated tasks, assignments, route/reassignment.
- `integration/travel.py`: converts learned/baseline times to an hourly oracle.
- `eval/experiments.py`: five-seed experiments and plots.
- `tests/`: normal, boundary, and impossible cases.
- `data_dictionary.csv`: source, derived, and generated columns.
