# Delivery routing with learned travel times

This project uses real September 2026 NYC TLC trip records to learn zone-to-zone
travel times. A small, explicitly synthetic delivery scenario then demonstrates
van assignment with a constraint-satisfaction problem (CSP) and stop ordering
with A* search.

> **Important distinction:** the TLC files contain completed passenger trips.
> They do not contain courier orders, delivery windows, van capacities, or a
> delivery depot. The neural network is trained and tested only on real TLC
> trips. The order and fleet inputs used by the dispatcher demo and CSP
> experiments are generated planning scenarios, clearly marked as synthetic.
> Taxi trip times approximate van travel times and are not courier performance
> measurements.

## Data in this workspace

| File | Origin | Rows | Use |
|---|---|---:|---|
| `yellow_tripdata_2026-09.parquet` | Official TLC September 2026 Yellow Taxi file | 3,876,044 | Real trip-duration observations |
| `green_tripdata_2026-09.parquet` | Official TLC September 2026 Green Taxi file | 46,676 | Real trip-duration observations |
| `fhvhv_tripdata_2026-09.parquet` | Supplied September 2026 FHVHV file | 20,987,681 | Real trip-duration observations |
| `taxi_zone_lookup.csv` | TLC zone lookup | 265 zones | Real zone IDs and names |

Source catalog: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>.
The monthly files are hosted at TLC's public CloudFront data URLs. The data are
used as supplied; TLC cautions that it cannot guarantee their accuracy.
`data/make_data.py` downloads a missing September file from the same TLC public
storage path over HTTPS. The complete raw column descriptions and all derived
and generated fields are in [data_dictionary.csv](./data_dictionary.csv).

## Setup

The project was verified with Python 3.15.2. Create and activate a virtual
environment, then install the pinned packages:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run commands

Prepare the reproducible 300,000-row sample, train the neural network and
baselines, and run the integrated demo:

```powershell
python data\make_data.py --seed 42 --max-rows 300000
python -m nn.train --seed 42
python run_demo.py --seed 42
```

`python run_demo.py` also prepares missing processed data/model files
automatically, then opens a browser-based presentation with chapters for the
trip model, generated order scenario, CSP assignment, an A*/UCS/Greedy route
search comparison, and the final route schedule. The search comparison holds
the van assignments fixed and shows feasibility, total estimated route cost,
and expanded/generated route states. Use the arrow keys or chapter buttons to
browse. The standalone HTML is saved
under `outputs\demo_presentation_seed42_nn.html`; `--no-open` generates it
without opening a browser. Training also supports direct script execution:

To present a tighter-window stress case that exercises CSP backtracking and
route-triggered reassignment, run:

```powershell
python run_demo.py --scenario stress
```

This uses reproducible seed 94 by default and records each assignment/replanning
attempt in the presentation and plan JSON. Pass `--seed` to explore a different
generated stress scenario.

To show an intentionally impossible capacity case with a clear no-solution
message, run:

```powershell
python run_demo.py --scenario impossible
```

The generated order requires more capacity than any demo van has. The CSP
rejects it; it does not silently change the order or vehicle capacity.

To show an order that is feasible only if its time window is ignored, run:

```powershell
python run_demo.py --scenario outside-window
```

The van starts at 08:00, but the order's window closes at 07:59. The planner
reports no solution and suggests extending the window rather than delivering
late.

```powershell
python nn\train.py
```

To compare the learned travel times with the non-NN baselines (each command
generates and opens the corresponding presentation):

```powershell
python run_demo.py --time-model nn
python run_demo.py --time-model historical
python run_demo.py --time-model distance
```

Show search counters and the CSP pruning variants on a repeatable small
instance:

```powershell
python -m eval.trace_demo
```

The trace output compares plain backtracking with backtracking plus AC-3
(forward checking disabled), and also shows the forward-checking variants on
the same generated instance. AC-3 is a domain-propagation technique used with
backtracking, not a separate search algorithm.

Run the three experiments (five seeds each), tests, or individual validation:

```powershell
python -m eval.experiments
python generate_report.py
python -m pytest -q
```

`python generate_report.py` creates `report.md` and four charts under
`outputs\report_figures\` from the saved model metrics and experiment results.

Processed data and model artifacts go under `data\processed\` and `artifacts\`.
The generated demo orders, plan JSON, integrated zone/hour matrix, and plot are
written under `outputs\`. Experiment CSV summaries and plots go under
`outputs\experiments\`.

## What the code does

1. **Preprocessing:** reads only the needed Parquet columns in batches; keeps
   weekday trips from September 2026 with known zones, positive distance, and
   durations from 1 to 120 minutes. Training uses September 1–23; testing uses
   September 24–30. Busiest zones and mean zone-pair distance are calculated
   from training days only. A seeded reservoir sample caps the result at
   approximately 300,000 real trips.
2. **Neural network:** an `MLPRegressor` receives pickup zone, drop-off zone,
   pickup hour, weekday, and training-only mean pair distance. The target is
   observed trip duration in minutes. Test metrics are compared with a
   training-only historical zone-pair average and a distance-only estimate.
3. **CSP assignment:** each generated order is a variable whose values are
   vans. Unary constraints check depot reachability and whether the order fits
   any vehicle. Binary constraints require different vans when neither
   ordering of two orders can meet both time windows. Aggregate van capacity
   is checked during assignment. Backtracking, forward checking, and AC-3 can
   be switched independently.
4. **Route search:** once assigned, each van's state is its current stop, set
   of completed orders, and current time. Actions visit an unvisited order.
   Cost includes predicted travel, waiting, and generated service time.
   A* uses a lower bound formed by summing the cheapest possible incoming
   travel time for each remaining order. UCS and Greedy are implemented for
   comparison. Search reports expanded/generated states and explicit failure
   reasons.
5. **Integration:** CSP assigns orders, A* orders each van's stops, and if a
   route reveals a late pair, the pair is added as a different-van constraint
   and assignment is retried up to five times. When no plan is found, the demo
   reports a reason and offers relevant time-window extensions in 15-minute
   steps; it does not silently change a window.

See [REVISION.md](./REVISION.md) for term definitions, algorithms,
worked logic, and the limitations to remember when presenting this project.

## Experiments and validation

- **Search:** A* vs UCS vs Greedy for 6–10 generated stops, with five fixed
  seeds; the six-stop A* result is checked against exhaustive permutations.
- **CSP:** BT vs BT+FC vs BT+FC+AC-3 for 15, 20, and 30 generated orders, with
  five fixed seeds and counters for assignments, backtracks, nodes, and domain
  values pruned. A 100,000-node cap distinguishes an inconclusive search from
  a proven no-solution result.
- **Replay:** NN vs historical-pair average vs distance-only on held-out real
  trips, evaluated on five seeded samples. Since TLC has no actual delivery
  deadlines, the displayed lateness measure is explicitly a **proxy**:
  predicted trip time greater than the observed trip duration plus a
  generated five-minute allowance. It is not observed courier lateness.
- **Tests:** 26 tests cover normal behavior, edge cases, and clear impossible
  cases. Algorithm fixtures are small and hand-constructed for known answers;
  they are not added to the real training dataset.

Detailed experiment results are saved under
[`outputs/experiments/`](./outputs/experiments/). The 10-minute talk and
5-minute live-demo outline is in [PRESENTATION.md](./PRESENTATION.md).

## Known limitations and remaining human-specific items

- TLC rides are not delivery jobs. No conclusion here should be presented as
  measured courier performance.
- The generated orders, time windows, service times, depot, vehicle count, and
  capacities are scenario assumptions, not facts from TLC.
- Pickup-to-drop-off zone trips are a proxy for zone-to-zone travel. Taxi
  durations can differ from van travel times and are subject to TLC data
  quality limitations.
- The current workspace has no Git repository and no team/member information.
  Therefore a truthful `git log` and individual contribution statements cannot
  be generated here; do not invent them. Add them when the project is in the
  group's repository and the authors confirm their contributions.
