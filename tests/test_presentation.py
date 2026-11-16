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
        "csp_stats": {
            "nodes": 3,
            "assignments": 1,
            "backtracks": 0,
            "values_pruned": 2,
        },
        "search_comparison": [
            {
                "algorithm": "A*",
                "feasible": True,
                "route_cost_minutes": 40.0,
                "nodes_expanded": 3,
                "nodes_generated": 5,
            },
            {
                "algorithm": "UCS",
                "feasible": True,
                "route_cost_minutes": 40.0,
                "nodes_expanded": 4,
                "nodes_generated": 6,
            },
            {
                "algorithm": "Greedy",
                "feasible": True,
                "route_cost_minutes": 45.0,
                "nodes_expanded": 2,
                "nodes_generated": 3,
            },
        ],
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

    assert "The full solution in five steps" in html
    assert "Prepare trip data" in html
    assert "Predict travel minutes" in html
    assert "CSP picks vans" in html
    assert "A* plans stops" in html
    assert "Repair conflicts" in html
    assert "The NN learns patterns from the training trips" in html
    assert "Generated order scenario" in html
    assert "Step 4 · CSP assigns orders to vans" in html
    assert "Search comparison" in html
    assert "Step 5 · Compare route-search methods" in html
    assert "same final van assignments" in html
    assert "A* · cost so far + estimate" in html
    assert "UCS · cost so far" in html
    assert "Greedy · estimate first" in html
    assert "45.0 min" in html
    assert "Step 6 · Read the route schedule" in html
    assert "MAE" in html
    assert "RMSE" in html
    assert "Van options removed" in html
    assert "One node means the solver examined one partial van plan." in html
    assert "Assignment and route-replanning attempts" in html
    assert "O01 ↔ O02" in html
    assert "2 backtracks summed across all assignment attempts." in html
    assert "09:00" in html
    assert "Presenter notes" not in html
    assert 'class="speaker"' not in html
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
