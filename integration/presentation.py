from __future__ import annotations

from html import escape


MODEL_LABELS = {
    "nn": "Neural network",
    "historical": "Historical zone-pair average",
    "distance": "Distance-only estimate",
}


def _clock(minutes: float) -> str:
    value = int(round(minutes))
    return f"{value // 60:02d}:{value % 60:02d}"


def _model_comparison(metrics: dict, selected_model: str) -> str:
    models = (
        ("nn", "Neural network"),
        ("historical_pair_average", "Historical pair average"),
        ("distance_only", "Distance-only"),
    )
    max_mae = max(
        float(metrics[key]["mae_minutes"])
        for key, _ in models
    )
    rows = []
    selected_key = {
        "nn": "nn",
        "historical": "historical_pair_average",
        "distance": "distance_only",
    }[selected_model]
    for key, label in models:
        mae = float(metrics[key]["mae_minutes"])
        rmse = float(metrics[key]["rmse_minutes"])
        width = max(3.0, mae / max_mae * 100.0)
        active = " selected" if key == selected_key else ""
        rows.append(
            f'<div class="metric-row{active}">'
            f'<div class="metric-label"><span>{escape(label)}</span>'
            f'<b>{mae:.2f} min MAE</b></div>'
            f'<div class="bar-track"><div class="bar" style="width:{width:.1f}%">'
            f'</div></div><small>RMSE {rmse:.2f} min</small></div>'
        )
    return "\n".join(rows)


def _schedule_svg(vans: list[dict]) -> str:
    all_orders = [
        order
        for van in vans
        for order in van["route"]
    ]
    if not all_orders:
        return '<p class="empty">No route schedule is available for this scenario.</p>'

    start = min(8 * 60, min(int(order["window_start"]) for order in all_orders))
    end = max(17 * 60, max(int(order["window_end"]) for order in all_orders))
    width = 1120
    left = 120
    right = 22
    chart_width = width - left - right
    rows = []
    y = 44
    for van in vans:
        rows.append(
            f'<text class="van-label" x="8" y="{y + 5}">'
            f'{escape(van["van_id"])}</text>'
        )
        for order in van["route"]:
            y += 36
            x1 = left + (int(order["window_start"]) - start) / (end - start) * chart_width
            x2 = left + (int(order["window_end"]) - start) / (end - start) * chart_width
            eta = float(order["arrival"])
            dot_x = left + (eta - start) / (end - start) * chart_width
            within_window = (
                int(order["window_start"]) <= eta <= int(order["window_end"])
            )
            dot_class = "arrival-ok" if within_window else "arrival-outside"
            rows.append(
                f'<text class="order-label" x="8" y="{y + 4}">'
                f'{escape(order["order_id"])}</text>'
                f'<rect class="window-bar" x="{x1:.1f}" y="{y - 8}" '
                f'width="{max(2.0, x2 - x1):.1f}" height="16" rx="5"/>'
                f'<circle class="{dot_class}" cx="{dot_x:.1f}" cy="{y}" r="6">'
                f'<title>{escape(order["order_id"])} predicted arrival '
                f'{_clock(eta)}</title></circle>'
                f'<text class="eta-label" x="{dot_x + 8:.1f}" y="{y + 4}">'
                f'{_clock(eta)}</text>'
            )
        y += 30

    ticks = []
    first_hour = (start + 59) // 60
    last_hour = end // 60
    for hour in range(first_hour, last_hour + 1):
        minute = hour * 60
        x = left + (minute - start) / (end - start) * chart_width
        ticks.append(
            f'<line class="axis-line" x1="{x:.1f}" y1="30" x2="{x:.1f}" '
            f'y2="{y}"/><text class="tick-label" x="{x:.1f}" y="20" '
            f'text-anchor="middle">{hour:02d}:00</text>'
        )
    return (
        f'<svg class="schedule" viewBox="0 0 {width} {y + 12}" role="img" '
        f'aria-label="Van route arrival times and order windows">'
        f'{"".join(ticks)}{"".join(rows)}</svg>'
    )


def render_presentation(data: dict) -> str:
    metrics = data["metrics"]
    selected_model = data["time_model"]
    model_label = MODEL_LABELS[selected_model]
    scenario = data.get("scenario", "standard")
    attempts = data.get("planning_attempts", [])
    total_backtracks = sum(
        int(attempt["csp_stats"]["backtracks"]) for attempt in attempts
    )
    orders = data["orders"]
    vans = data["vans"]
    feature_names = {
        "pickup_zone": "Pickup zone",
        "dropoff_zone": "Drop-off zone",
        "hour": "Pickup hour",
        "weekday": "Weekday",
        "mean_pair_distance": "Training-only average distance for the zone pair",
    }
    features = ", ".join(
        feature_names.get(feature, feature.replace("_", " "))
        for feature in metrics["features"]
    )
    orders_html = "".join(
        "<tr>"
        f'<td><b>{escape(order["order_id"])}</b></td>'
        f'<td>{escape(order["zone_name"])} <span class="muted">'
        f'(zone {order["zone"]})</span></td>'
        f'<td>{_clock(order["window_start"])}–{_clock(order["window_end"])}</td>'
        f'<td>{order["demand"]}</td><td>{order["service_minutes"]:.0f} min</td>'
        "</tr>"
        for order in orders
    )
    van_cards = "".join(
        f'<article class="van-card"><div class="van-card-top">'
        f'<h3>{escape(van["van_id"])}</h3>'
        f'<span>{van["demand"]} / {van["capacity"]} capacity used</span></div>'
        f'<div class="capacity-track"><div style="width:'
        f'{100 * van["demand"] / max(1, van["capacity"]):.1f}%"></div></div>'
        f'<p><b>Assigned orders:</b> '
        f'{", ".join(escape(order_id) for order_id in van["order_ids"])}</p>'
        f'<p class="muted">A* stop sequence: '
        f'{" → ".join(escape(order["order_id"]) for order in van["route"])}</p>'
        f'<small>{van["expanded"]} states expanded · '
        f'{van["generated"]} generated · route cost {van["cost"]:.1f} min</small>'
        f'</article>'
        for van in vans
    )
    csp = data["csp_stats"]
    attempt_rows = "".join(
        "<tr>"
        f'<td>{attempt["number"]}</td>'
        f'<td>{attempt["csp_stats"]["nodes"]}</td>'
        f'<td>{attempt["csp_stats"]["assignments"]}</td>'
        f'<td>{attempt["csp_stats"]["backtracks"]}</td>'
        f'<td>{escape("Feasible" if attempt["route_feasible"] else "Retry required" if attempt["route_feasible"] is False else "Assignment blocked")}</td>'
        f'<td>{escape(" ↔ ".join(attempt["blocked_pair"]) if attempt["blocked_pair"] else "—")}</td>'
        "</tr>"
        for attempt in attempts
    )
    scenario_label = "Stress case" if scenario == "stress" else "Standard case"
    scenario_explanation = (
        "Tighter seeded windows intentionally trigger assignment backtracking "
        "and route-conflict reassignment before a feasible plan is found."
        if scenario == "stress" else
        "Wide seeded windows make this a straightforward baseline walkthrough."
    )
    feasible = data["feasible"]
    status_text = "Feasible plan found" if feasible else "No feasible plan found"
    plan_notice = (
        '<p class="callout success">The generated scenario was assigned and '
        'routed successfully.</p>'
        if feasible else
        f'<p class="callout warning"><b>{escape(data["reason"] or status_text)}</b>'
        f'<br>Review the generated order windows or vehicle capacity. This is a '
        f'scenario-planning outcome, not a real-world delivery result.</p>'
    )
    suggestions = "".join(
        f"<li>{escape(suggestion)}</li>" for suggestion in data["suggestions"]
    )
    schedule = _schedule_svg(vans) if feasible else (
        '<p class="empty">A route timeline is shown only when the scenario has '
        'a feasible assignment and route.</p>'
    )
    test_period = escape(str(metrics["test_period"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trip planning demo · presentation</title>
<style>
:root{{--ink:#172b3a;--muted:#607482;--blue:#276b8e;--pale:#eaf3f7;
--line:#d6e1e7;--green:#14805e;--orange:#c25332;--paper:#f4f7f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.5 "Segoe UI",Arial,sans-serif}}button{{font:inherit}}
.topbar{{background:#102a3a;color:white;padding:13px max(24px,calc((100vw - 1180px)/2));
display:flex;justify-content:space-between;align-items:center;gap:14px}}
.brand{{font-weight:750;letter-spacing:.03em}}.top-meta{{color:#c4d6df;font-size:14px}}
main{{max-width:1180px;margin:26px auto 110px;padding:0 24px}}
.hero{{padding:30px 34px;border-radius:20px;background:linear-gradient(120deg,#14394f,#236b8d);
color:white;box-shadow:0 12px 32px #193e5220}}.eyebrow{{text-transform:uppercase;
letter-spacing:.12em;font-size:12px;font-weight:750;color:#add8e7}}
h1{{font-size:clamp(30px,5vw,48px);line-height:1.08;margin:8px 0 12px}}
.hero p{{max-width:820px;color:#e3eff4;margin:0}}.badge{{display:inline-flex;
padding:5px 10px;border-radius:999px;background:#ffffff20;font-size:13px}}
.nav{{display:flex;gap:8px;margin:20px 0 0;overflow:auto}}
.nav button,.controls button{{border:1px solid var(--line);border-radius:10px;
padding:9px 14px;background:white;color:var(--ink);cursor:pointer;white-space:nowrap}}
.nav button.active{{background:var(--blue);color:white;border-color:var(--blue)}}
.slide{{display:none;background:white;border:1px solid var(--line);border-radius:18px;
padding:28px 32px;margin-top:14px;box-shadow:0 5px 18px #18394a0c}}
.slide.active{{display:block}}.slide h2{{font-size:28px;margin:0 0 6px}}
.lead{{font-size:18px;color:var(--muted);max-width:900px;margin-top:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0}}
.card,.van-card{{border:1px solid var(--line);border-radius:14px;padding:17px;background:#fff}}
.card h3,.van-card h3{{margin:0 0 5px}}.big{{font-size:30px;font-weight:780;letter-spacing:-.03em}}
.muted,small{{color:var(--muted)}}small{{display:block;margin-top:8px}}
.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:22px}}
.flow article{{padding:18px;border:1px solid var(--line);border-radius:14px;position:relative}}
.flow article:not(:last-child):after{{content:"→";position:absolute;right:-21px;top:38%;
color:var(--blue);font-size:24px;font-weight:700;z-index:1}}
.step{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.1em}}
.callout{{padding:14px 16px;border-radius:12px;background:var(--pale);margin:18px 0}}
.success{{background:#e8f6f0;color:#145c47}}.warning{{background:#fff1e8;color:#8e3b24}}
.metric-row{{margin:15px 0;padding:10px;border-radius:10px}}
.metric-row.selected{{background:#eff7fa}}.metric-label{{display:flex;justify-content:space-between;gap:16px}}
.bar-track,.capacity-track{{height:10px;background:#e9eff2;border-radius:20px;overflow:hidden;margin:6px 0}}
.bar,.capacity-track div{{height:100%;background:#428daf;border-radius:20px}}
.metric-row.selected .bar{{background:#15805e}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;margin-top:18px}}
table{{border-collapse:collapse;width:100%;min-width:720px}}th,td{{padding:11px 13px;
text-align:left;border-bottom:1px solid var(--line)}}th{{background:#eff5f7;font-size:13px}}
tr:last-child td{{border-bottom:0}}.van-grid{{display:grid;grid-template-columns:repeat(auto-fit,
minmax(260px,1fr));gap:14px;margin-top:18px}}.van-card-top{{display:flex;
justify-content:space-between;gap:8px;align-items:center}}.van-card-top span{{font-size:13px;color:var(--muted)}}
.capacity-track div{{background:#61a989}}.schedule-wrap{{overflow-x:auto;border:1px solid var(--line);
border-radius:12px;padding:12px;margin-top:15px}}.schedule{{width:100%;min-width:800px;height:auto}}
.axis-line{{stroke:#e2e9ed;stroke-width:1}}.tick-label,.eta-label{{fill:#526875;font-size:12px}}
.order-label{{fill:#172b3a;font-size:13px;font-weight:700}}.van-label{{fill:#276b8e;
font-size:14px;font-weight:800}}.window-bar{{fill:#acd4e4;stroke:#428daf}}
.arrival-ok{{fill:#16845e;stroke:white;stroke-width:2}}.arrival-outside{{fill:#c25332;
stroke:white;stroke-width:2}}.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0;color:var(--muted);font-size:14px}}
.legend span:before{{content:"";display:inline-block;width:12px;height:12px;border-radius:50%;
margin-right:7px;vertical-align:-1px;background:#16845e}}.legend .bar-key:before{{
width:16px;height:10px;border-radius:4px;background:#acd4e4}}.legend .late-key:before{{background:#c25332}}
.speaker{{margin-top:20px;border-top:1px solid var(--line);padding-top:12px}}
.speaker summary{{cursor:pointer;color:var(--blue);font-weight:700}}.empty{{padding:22px;
border-radius:12px;background:#f4f7f8;color:var(--muted)}}
.controls{{position:fixed;bottom:0;left:0;right:0;background:#ffffffed;
backdrop-filter:blur(8px);border-top:1px solid var(--line);padding:12px max(24px,calc((100vw - 1180px)/2));
display:flex;justify-content:space-between;align-items:center;gap:12px}}
.controls .right{{display:flex;gap:8px}}.progress{{font-size:14px;color:var(--muted)}}
@media(max-width:700px){{main{{padding:0 14px;margin-top:14px}}.slide{{padding:20px 17px}}
.hero{{padding:24px 20px}}.flow{{grid-template-columns:1fr;gap:10px}}.flow article:after{{display:none}}
.top-meta{{display:none}}.controls{{padding:10px 14px}}}}
@media print{{.nav,.controls{{display:none}}body{{background:white}}main{{margin:0 auto}}
.slide{{display:block!important;page-break-after:always;box-shadow:none}}.hero{{print-color-adjust:exact}}}}
</style>
</head>
<body>
<header class="topbar"><div class="brand">CITY LOGISTICS · PROJECT WALKTHROUGH</div>
<div class="top-meta">Seed {data["seed"]} · Active travel-time method: {escape(model_label)}</div></header>
<main>
<section class="hero">
<span class="badge">PRESENTATION MODE · 5 CHAPTERS · {escape(scenario_label.upper())}</span>
<h1>From real trip data<br>to a delivery plan</h1>
<p>{escape(scenario_explanation)} A guided demo of how travel-time estimates, constraint-based vehicle assignment,
and A* route search work together. Use the chapter buttons or arrow keys to present.</p>
<p style="margin-top:16px"><span class="badge">Real TLC trip records for model training</span>
&nbsp;<span class="badge">Generated orders for the planning scenario</span></p>
</section>
<nav class="nav" aria-label="Presentation chapters">
<button class="active" data-slide="0">1 · Overview</button>
<button data-slide="1">2 · Data &amp; model</button>
<button data-slide="2">3 · Scenario</button>
<button data-slide="3">4 · Assignment</button>
<button data-slide="4">5 · Routes &amp; limits</button>
</nav>
<section class="slide active" aria-label="Overview">
<h2>One planning problem, three connected parts</h2>
<p class="lead">Estimate travel time from past trips, assign orders to vehicles under constraints,
then find a stop order that respects the order windows.</p>
<div class="flow">
<article><span class="step">01 · ESTIMATE</span><h3>Travel-time model</h3>
<p>Uses trip history to estimate zone-to-zone travel duration for the selected hour.</p></article>
<article><span class="step">02 · ASSIGN</span><h3>CSP vehicle allocation</h3>
<p>Chooses a van per order while respecting time conflicts and each van's capacity.</p></article>
<article><span class="step">03 · ORDER</span><h3>A* route search</h3>
<p>Searches for the sequence of assigned stops with travel, waiting, and service time.</p></article>
</div>
<div class="grid">
<div class="card"><div class="muted">Test-set mean absolute error</div>
<div class="big">{float(metrics["nn"]["mae_minutes"]):.2f} min</div>
<small>Neural network prediction error on unseen real trips</small></div>
<div class="card"><div class="muted">Scenario orders</div><div class="big">{len(orders)}</div>
<small>Generated planning inputs, not TLC delivery records</small></div>
<div class="card"><div class="muted">Plan status</div><div class="big" style="font-size:24px">{escape(status_text)}</div>
<small>Computed from seed {data["seed"]} with {escape(model_label.lower())}</small></div>
</div>
<div class="callout"><b>Important scope note:</b> real taxi trips train and test the travel-time model.
Delivery orders, their windows, service times, depot, and fleet capacities are generated assumptions.</div>
<details class="speaker"><summary>Presenter notes</summary><p>Start with the three-stage story.
Emphasize that the model is evaluated on actual held-out trip durations, while the
downstream delivery-planning inputs are synthetic because the taxi data has no courier orders.</p></details>
</section>
<section class="slide" aria-label="Data and model">
<h2>What the model learns</h2>
<p class="lead">It predicts a trip's duration in minutes, not a delivery success category.
The displayed accuracy is therefore an error in minutes rather than a percentage.</p>
<div class="grid">
<div class="card"><div class="muted">Training examples</div><div class="big">{int(metrics["train_rows"]):,}</div>
<small>Real trips · September 1–23, 2026</small></div>
<div class="card"><div class="muted">Held-out test examples</div><div class="big">{int(metrics["test_rows"]):,}</div>
<small>Real trips · {test_period}</small></div>
<div class="card"><div class="muted">Model</div><div class="big">32 → 16</div>
<small>Hidden-layer units · ReLU · Adam · early stopping · {int(metrics["neural_network_iterations"])} iterations</small></div>
</div>
<h3>Test error comparison</h3>
<p class="muted">Lower MAE is better. Each estimate is compared with the observed duration on the same held-out trips.</p>
{_model_comparison(metrics, selected_model)}
<div class="callout"><b>Model inputs:</b> {escape(features)}.<br>
The neural network's test MAE is {float(metrics["nn"]["mae_minutes"]):.2f} minutes;
its RMSE is {float(metrics["nn"]["rmse_minutes"]):.2f} minutes. RMSE penalizes larger misses more.</div>
<details class="speaker"><summary>Presenter notes</summary><p>Explain the chronological split:
train on earlier September weekdays and evaluate on the later weekdays, reducing the risk
of testing on the same time period used to fit the model. The baselines show whether the
network adds value beyond simple averages.</p></details>
</section>
<section class="slide" aria-label="Generated order scenario">
<h2>Scenario input: generated orders</h2>
<p class="lead">These are presentation/demo inputs. Zone IDs correspond to NYC taxi zones,
but order demand, service time, and time windows are generated for this scenario.</p>
<div class="grid">
<div class="card"><div class="muted">Order count</div><div class="big">{len(orders)}</div></div>
<div class="card"><div class="muted">Depot zone</div><div class="big">{data["depot_zone"]}</div>
<small>{escape(data["depot_name"])}</small></div>
<div class="card"><div class="muted">Fleet</div><div class="big">{len(vans)} vans</div>
<small>Capacity {vans[0]["capacity"] if vans else 0} units per van</small></div>
</div>
<div class="callout">Each order has a destination zone, an allowed time window, demand,
and a small generated service duration. Order IDs are synthetic labels.</div>
<div class="table-wrap"><table><thead><tr><th>Order</th><th>Destination zone</th>
<th>Allowed window</th><th>Demand</th><th>Service time</th></tr></thead>
<tbody>{orders_html}</tbody></table></div>
<details class="speaker"><summary>Presenter notes</summary><p>Point out one example order:
its time window is a constraint on when the van can serve it, and demand consumes vehicle capacity.
Be explicit that this scenario is generated rather than observed courier data.</p></details>
</section>
<section class="slide" aria-label="Vehicle assignment">
<h2>Constraint-based assignment</h2>
<p class="lead">The CSP selects a van for each order. Capacity and pairwise time-window
conflicts constrain which combinations are allowed.</p>
{plan_notice}
<div class="grid">
<div class="card"><div class="muted">Search nodes</div><div class="big">{int(csp.get("nodes", 0)):,}</div></div>
<div class="card"><div class="muted">Assignments tried</div><div class="big">{int(csp.get("assignments", 0)):,}</div></div>
<div class="card"><div class="muted">Backtracks</div><div class="big">{int(csp.get("backtracks", 0)):,}</div>
<small>{total_backtracks} total across all assignment attempts.</small></div>
<div class="card"><div class="muted">Route-triggered reassignments</div><div class="big">{int(data["reassignments"])}</div>
<small>Late route conflicts can be added as different-van constraints.</small></div>
</div>
<h3>Assignment and route-replanning cycles</h3>
<p class="muted">Each retry adds a route-discovered order conflict to the CSP, then assigns and searches again.</p>
<div class="table-wrap"><table><thead><tr><th>Attempt</th><th>CSP nodes</th>
<th>Choices tried</th><th>Backtracks</th><th>Outcome</th><th>Route conflict added</th></tr></thead>
<tbody>{attempt_rows or '<tr><td colspan="6">No planning attempts recorded.</td></tr>'}</tbody></table></div>
<div class="van-grid">{van_cards or '<p class="empty">No vehicle assignment is available.</p>'}</div>
{f'<div class="callout"><b>Possible adjustments:</b><ul>{suggestions}</ul></div>' if suggestions else ''}
<details class="speaker"><summary>Presenter notes</summary><p>Describe each order as a variable
whose domain is the set of vans. CSP constraints remove assignments that violate capacity
or a pairwise scheduling conflict. A backtrack means the solver undoes a choice and tries another.</p></details>
</section>
<section class="slide" aria-label="Routes and limitations">
<h2>Route schedule and what it means</h2>
<p class="lead">After assignment, A* chooses an order of stops for each van.
The chart compares predicted arrival times with the generated order windows.</p>
<div class="legend"><span class="bar-key">Allowed order window</span>
<span>Predicted arrival inside window</span><span class="late-key">Arrival outside window</span></div>
<div class="schedule-wrap">{schedule}</div>
<div class="grid">
<div class="card"><h3>Search</h3><p>Route state tracks the current stop, completed orders, and time.
The objective includes travel, waiting, and service duration.</p></div>
<div class="card"><h3>Travel-time caveat</h3><p>TLC taxi trip durations approximate travel between zones;
they are not measured courier-van times.</p></div>
<div class="card"><h3>Evaluation caveat</h3><p>The model's MAE/RMSE evaluate held-out real taxi trips.
They do not measure actual courier lateness or delivery performance.</p></div>
</div>
<div class="callout">Route arrivals are predictions based on the selected time model.
The blue bars and scenario inputs are generated; they should not be presented as real delivery outcomes.</div>
<details class="speaker"><summary>Presenter notes</summary><p>Read the chart as a timeline:
blue is the allowed window and the dot is the estimated arrival. A* searches feasible
stop sequences using estimated travel times. Close by distinguishing model accuracy on taxi
trips from delivery performance, which this dataset cannot establish.</p></details>
</section>
</main>
<footer class="controls"><div class="progress" id="progress">Chapter 1 of 5</div>
<div class="right"><button id="prev" type="button">← Previous</button>
<button id="next" type="button">Next →</button><button id="full" type="button">Present full screen</button></div></footer>
<script>
const slides=[...document.querySelectorAll(".slide")];
const tabs=[...document.querySelectorAll(".nav button")];
let current=0;
function show(index){{current=Math.max(0,Math.min(slides.length-1,index));
slides.forEach((s,i)=>s.classList.toggle("active",i===current));
tabs.forEach((b,i)=>b.classList.toggle("active",i===current));
document.getElementById("progress").textContent=`Chapter ${{current+1}} of ${{slides.length}}`;
document.getElementById("prev").disabled=current===0;
document.getElementById("next").disabled=current===slides.length-1;
window.scrollTo({{top:0,behavior:"smooth"}});}}
tabs.forEach((button,index)=>button.addEventListener("click",()=>show(index)));
document.getElementById("prev").addEventListener("click",()=>show(current-1));
document.getElementById("next").addEventListener("click",()=>show(current+1));
document.getElementById("full").addEventListener("click",()=>{{
if(document.fullscreenElement) document.exitFullscreen();
else document.documentElement.requestFullscreen?.();
}});
document.addEventListener("keydown",event=>{{
if(event.key==="ArrowRight"||event.key==="PageDown") show(current+1);
if(event.key==="ArrowLeft"||event.key==="PageUp") show(current-1);
}});
show(0);
</script>
</body>
</html>"""
