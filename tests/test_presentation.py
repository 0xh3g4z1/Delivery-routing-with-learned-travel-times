from integration.presentation import render_presentation


def test_presentation_contains_walkthrough_and_escapes_zone_names():
    html = render_presentation({
        "seed": 42,
        "time_model": "nn",
        "metrics": {
            "train_rows": 100,
            "test_rows": 20,
            "test_period": "Sep 24–30",
            "features": ["pickup_zone", "mean_pair_distance"],
            "neural_network_iterations": 4,
            "nn": {"mae_minutes": 5.0, "rmse_minutes": 7.0},
            "historical_pair_average": {"mae_minutes": 6.0, "rmse_minutes": 8.0},
            "distance_only": {"mae_minutes": 10.0, "rmse_minutes": 12.0},
        },
        "orders": [{
            "order_id": "O01",
            "zone": 10,
            "zone_name": '<script>alert("x")</script>',
            "window_start": 500,
            "window_end": 600,
            "demand": 2,
            "service_minutes": 5.0,
        }],
        "depot_zone": 10,
        "depot_name": "Test depot",
        "vans": [{
            "van_id": "Van 1",
            "capacity": 10,
            "demand": 2,
            "order_ids": ["O01"],
            "route": [{
                "order_id": "O01",
                "window_start": 500,
                "window_end": 600,
                "arrival": 540.0,
            }],
            "cost": 40.0,
            "expanded": 3,
            "generated": 5,
        }],
        "csp_stats": {"nodes": 3, "assignments": 1, "backtracks": 0},
        "planning_attempts": [
            {
                "number": 1,
                "csp_stats": {
                    "nodes": 4,
                    "assignments": 5,
                    "backtracks": 2,
                },
                "route_feasible": False,
                "blocked_pair": ("O01", "O02"),
                "reason": "No feasible route",
            },
            {
                "number": 2,
                "csp_stats": {
                    "nodes": 3,
                    "assignments": 2,
                    "backtracks": 0,
                },
                "route_feasible": True,
                "blocked_pair": None,
                "reason": None,
            },
        ],
        "reassignments": 0,
        "feasible": True,
        "reason": None,
        "suggestions": [],
    })

    assert "Data &amp; model" in html
    assert "Generated order scenario" in html
    assert "Constraint-based assignment" in html
    assert "Route schedule and what it means" in html
    assert "MAE" in html
    assert "Assignment and route-replanning cycles" in html
    assert "O01 ↔ O02" in html
    assert "2 total across all assignment attempts" in html
    assert "09:00" in html
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
