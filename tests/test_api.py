import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.check_plan import check_plan

client = TestClient(app)

VALID_PAYLOAD = {
    "stock": {"width": 900, "height": 600},
    "kerf": 3,
    "items": [
        {"id": "flyer", "width": 210, "height": 148, "quantity": 8},
        {"id": "card", "width": 90, "height": 55, "quantity": 20},
    ],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_plan_happy_path():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["stock"] == {"width": 900.0, "height": 600.0}
    assert body["kerf"] == 3.0
    assert body["summary"]["piece_count"] == 28
    assert body["summary"]["sheet_count"] == len(body["sheets"])
    check_plan(body)


def test_create_plan_auto_generates_item_ids():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"width": 50, "height": 50, "quantity": 2}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    items = {p["item"] for s in response.json()["sheets"] for p in s["placements"]}
    assert items == {"item-1"}


def test_unfit_item_returns_clear_error():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "poster", "width": 150, "height": 80, "quantity": 1}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "poster" in detail
    assert "150" in detail and "100" in detail


def test_invalid_stock_dimensions_rejected():
    payload = {"stock": {"width": -10, "height": 100},
               "items": [{"width": 10, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422

    payload["stock"] = {"width": 100, "height": 0}
    assert client.post("/plans", json=payload).status_code == 422


def test_invalid_item_dimensions_rejected():
    payload = {"stock": {"width": 100, "height": 100},
               "items": [{"width": 0, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_invalid_quantity_rejected():
    payload = {"stock": {"width": 100, "height": 100},
               "items": [{"width": 10, "height": 10, "quantity": 0}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_negative_kerf_rejected():
    payload = {"stock": {"width": 100, "height": 100}, "kerf": -1,
               "items": [{"width": 10, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_empty_item_list_rejected():
    payload = {"stock": {"width": 100, "height": 100}, "items": []}
    assert client.post("/plans", json=payload).status_code == 422


def test_duplicate_item_ids_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [
            {"id": "A", "width": 10, "height": 10, "quantity": 1},
            {"id": "A", "width": 20, "height": 20, "quantity": 1},
        ],
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_missing_fields_rejected():
    assert client.post("/plans", json={}).status_code == 422
    assert client.post("/plans", json={"stock": {"width": 100}}).status_code == 422


# ---------------------------------------------------------------------------
# locked placements
# ---------------------------------------------------------------------------

LOCKED_PAYLOAD = {
    "stock": {"width": 100, "height": 100},
    "kerf": 5,
    "items": [{"id": "A", "width": 45, "height": 40, "quantity": 3}],
    "locked_placements": [
        {"item": "A", "x": 0, "y": 0, "rotated": False},
        {"item": "A", "x": 50, "y": 0, "rotated": False},
    ],
}


def test_locked_placements_happy_path():
    response = client.post("/plans", json=LOCKED_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    sheet0 = body["sheets"][0]
    locked = [p for p in sheet0["placements"] if p["locked"]]
    unlocked = [p for p in sheet0["placements"] if not p["locked"]]
    assert len(locked) == 2
    assert {(p["x"], p["y"]) for p in locked} == {(0.0, 0.0), (50.0, 0.0)}
    # One remaining item of the quantity is still placed somewhere.
    assert body["summary"]["piece_count"] == 3
    assert len(locked) + len(unlocked) <= 3


def test_locked_defaults_rotated_to_false():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 0, "y": 0}],
    }
    body = client.post("/plans", json=payload).json()
    assert body["sheets"][0]["placements"][0]["rotated"] is False


def test_locked_placements_without_field_keep_working():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert all(p["locked"] is False
               for s in response.json()["sheets"] for p in s["placements"])


def test_locked_reference_to_unknown_item_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [{"item": "B", "x": 0, "y": 0}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "B" in response.json()["detail"]


def test_locked_count_exceeding_quantity_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 50},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#2" in response.json()["detail"]


def test_locked_overlap_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 30, "y": 30},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#1" in response.json()["detail"]
    assert "#2" in response.json()["detail"]


def test_locked_out_of_bounds_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 70, "y": 70}],
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_locked_non_guillotine_layout_rejected():
    # Four interlocking blades: non-slicing, even with zero kerf.
    coords = [(0, 0, 80, 120), (80, 0, 120, 80),
              (160, 80, 40, 120), (0, 120, 160, 80)]
    payload = {
        "stock": {"width": 200, "height": 200},
        "kerf": 0,
        "items": [
            {"id": f"p{i}", "width": w, "height": h, "quantity": 1}
            for i, (x, y, w, h) in enumerate(coords, 1)
        ],
        "locked_placements": [
            {"item": f"p{i}", "x": x, "y": y}
            for i, (x, y, w, h) in enumerate(coords, 1)
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "p1" in detail and "guillotine" in detail


def test_locked_touching_with_kerf_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 45, "y": 0},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#1" in response.json()["detail"]


def test_locked_placements_all_on_first_sheet():
    response = client.post("/plans", json=LOCKED_PAYLOAD)
    body = response.json()
    for index, sheet in enumerate(body["sheets"]):
        for p in sheet["placements"]:
            if p["locked"]:
                assert index == 0


def test_locked_negative_coordinates_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": -1, "y": 0}],
    }
    assert client.post("/plans", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# max_cuts_per_sheet
# ---------------------------------------------------------------------------

def test_max_cuts_per_sheet_limits_cuts():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 0,
        "items": [{"id": "A", "width": 50, "height": 100, "quantity": 4}],
        "max_cuts_per_sheet": 1,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["summary"]["sheet_count"] == 2
    assert all(s["cut_count"] <= 1 for s in body["sheets"])


def test_max_cuts_per_sheet_rejects_uncuttable_item():
    # A 40x40 piece on a blank 100x100 sheet (kerf 3) needs two cuts.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "max_cuts_per_sheet": 1,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]


def test_max_cuts_per_sheet_counts_locked_tree():
    # The locked pair needs three cuts on the first sheet.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 5,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 0},
        ],
        "max_cuts_per_sheet": 2,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]


def test_max_cuts_per_sheet_locked_within_limit():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 5,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 0},
        ],
        "max_cuts_per_sheet": 3,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["sheets"][0]["cut_count"] == 3


def test_max_cuts_per_sheet_validation():
    base = {
        "stock": {"width": 100, "height": 100},
        "items": [{"width": 10, "height": 10, "quantity": 1}],
    }
    for bad in (0, -1, 1.5, "two"):
        payload = {**base, "max_cuts_per_sheet": bad}
        assert client.post("/plans", json=payload).status_code == 422


def test_cut_count_returned_per_sheet():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)  # check_plan asserts cut_count matches the cut tree
    assert all("cut_count" in s for s in body["sheets"])


# ---------------------------------------------------------------------------
# gripper_zone (机台叼口保留区)
# ---------------------------------------------------------------------------

def test_gripper_zone_happy_path():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 4}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    root = body["sheets"][0]["cut_tree"]
    assert root["direction"] == "horizontal"
    strip = root["children"][0]
    assert strip["type"] == "waste"
    assert (strip["x"], strip["y"], strip["width"], strip["height"]) == (
        0.0, 0.0, 100.0, 10.0)
    for sheet in body["sheets"]:
        for p in sheet["placements"]:
            assert p["y"] >= 12 - 1e-6
        assert sheet["cut_count"] >= 1  # the separator cut


def test_gripper_zone_appears_on_every_sheet():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 8}],
        "gripper_zone": {"edge": "left", "size": 10},
    }
    body = client.post("/plans", json=payload).json()
    check_plan(body)
    assert len(body["sheets"]) >= 2
    for sheet in body["sheets"]:
        leaves = []
        stack = [sheet["cut_tree"]]
        while stack:
            node = stack.pop()
            if node["type"] == "cut":
                stack.extend(node["children"])
            else:
                leaves.append(node)
        assert any(
            n["type"] == "waste" and n["x"] == 0.0 and n["width"] == 10.0
            for n in leaves
        )
        for p in sheet["placements"]:
            assert p["x"] >= 12 - 1e-6


@pytest.mark.parametrize("edge", ["top", "right", "bottom", "left"])
def test_gripper_zone_all_edges_accepted(edge):
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 30, "height": 30, "quantity": 2}],
        "gripper_zone": {"edge": edge, "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    check_plan(response.json())


def test_gripper_zone_without_field_keeps_legacy_result():
    before = client.post("/plans", json=VALID_PAYLOAD).json()
    payload = {**VALID_PAYLOAD, "gripper_zone": None}
    after = client.post("/plans", json=payload).json()
    assert before == after


def test_gripper_zone_invalid_edge_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "gripper_zone": {"edge": "north", "size": 10},
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_gripper_zone_non_positive_size_rejected():
    for size in (0, -5):
        payload = {
            "stock": {"width": 100, "height": 100},
            "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
            "gripper_zone": {"edge": "top", "size": size},
        }
        assert client.post("/plans", json=payload).status_code == 422


def test_gripper_zone_leaves_no_usable_area_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 100},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "gripper_zone" in response.json()["detail"]


def test_gripper_zone_item_does_not_fit_inner_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "poster", "width": 95, "height": 95, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "poster" in detail and "gripper_zone" in detail


def test_gripper_zone_locked_entering_zone_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 0, "y": 5}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "#1" in detail and "A" in detail


def test_gripper_zone_locked_straddling_separator_rejected():
    # y=11 lands inside the kerf strip [10, 12).
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 0, "y": 11}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_gripper_zone_separator_counts_towards_cut_limit():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
        "max_cuts_per_sheet": 2,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]


def test_gripper_zone_with_locked_within_limit():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 12, "y": 12}],
        "gripper_zone": {"edge": "top", "size": 10},
        "max_cuts_per_sheet": 4,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    check_plan(response.json())


# ---------------------------------------------------------------------------
# bleed (四边出血)
# ---------------------------------------------------------------------------

BLEED_PAYLOAD = {
    "stock": {"width": 100, "height": 100},
    "kerf": 0,
    "items": [
        {"id": "A", "width": 40, "height": 40, "quantity": 1,
         "bleed": {"top": 5, "right": 5, "bottom": 5, "left": 5}},
    ],
}


def test_bleed_happy_path_returns_production_rect():
    response = client.post("/plans", json=BLEED_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    p = body["sheets"][0]["placements"][0]
    # Net coordinates are inset by the bleed from the production rect.
    assert (p["x"], p["y"], p["width"], p["height"]) == (5.0, 5.0, 40.0, 40.0)
    assert p["production_rect"] == {"x": 0.0, "y": 0.0,
                                    "width": 50.0, "height": 50.0}
    # The cut tree item leaf covers the production rectangle.
    leaf = body["sheets"][0]["cut_tree"]
    while leaf["type"] == "cut":
        leaf = leaf["children"][0]
    assert leaf["type"] == "item"
    assert (leaf["x"], leaf["y"], leaf["width"], leaf["height"]) == (
        0.0, 0.0, 50.0, 50.0)


def test_bleed_defaults_keep_response_compatible():
    body = client.post("/plans", json=VALID_PAYLOAD).json()
    assert all("production_rect" not in p
               for s in body["sheets"] for p in s["placements"])


def test_bleed_explicit_zero_keeps_response_compatible():
    payload = dict(VALID_PAYLOAD)
    payload["items"] = [
        {**item, "bleed": {"top": 0, "right": 0, "bottom": 0, "left": 0}}
        for item in VALID_PAYLOAD["items"]
    ]
    assert client.post("/plans", json=payload).json() == \
        client.post("/plans", json=VALID_PAYLOAD).json()


def test_bleed_explicit_null_keeps_response_compatible():
    payload = dict(VALID_PAYLOAD)
    payload["items"] = [{**item, "bleed": None} for item in VALID_PAYLOAD["items"]]
    assert client.post("/plans", json=payload).json() == \
        client.post("/plans", json=VALID_PAYLOAD).json()


def test_bleed_negative_side_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1,
                   "bleed": {"left": -1}}],
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_bleed_partial_sides_default_to_zero():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1,
                   "bleed": {"left": 5}}],
    }
    body = client.post("/plans", json=payload).json()
    check_plan(body)
    p = body["sheets"][0]["placements"][0]
    assert p["production_rect"] == {"x": 0.0, "y": 0.0,
                                    "width": 45.0, "height": 40.0}
    assert (p["x"], p["y"]) == (5.0, 0.0)


def test_bleed_utilization_counts_net_area_only():
    body = client.post("/plans", json=BLEED_PAYLOAD).json()
    sheet = body["sheets"][0]
    assert sheet["utilization"] == pytest.approx(1600 / 10000)
    assert body["summary"]["used_area"] == pytest.approx(1600)


def test_bleed_rotates_with_locked_piece():
    # Rotated 90° clockwise: sheet-frame bleed is top=left(4),
    # right=top(1), bottom=right(2), left=bottom(3).
    payload = {
        "stock": {"width": 300, "height": 300},
        "items": [{"id": "A", "width": 100, "height": 50, "quantity": 1,
                   "bleed": {"top": 1, "right": 2, "bottom": 3, "left": 4}}],
        "locked_placements": [{"item": "A", "x": 10, "y": 20,
                               "rotated": True}],
    }
    body = client.post("/plans", json=payload).json()
    check_plan(body)
    p = body["sheets"][0]["placements"][0]
    assert (p["x"], p["y"]) == (10.0, 20.0)  # net coords as requested
    assert (p["width"], p["height"]) == (50.0, 100.0)
    assert p["production_rect"] == {"x": 7.0, "y": 16.0,
                                    "width": 54.0, "height": 106.0}


def test_locked_bleed_outside_stock_rejected():
    # Net rect is inside the stock, but the left bleed crosses the edge.
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1,
                   "bleed": {"left": 5}}],
        "locked_placements": [{"item": "A", "x": 0, "y": 0}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#1" in response.json()["detail"]


def test_locked_bleed_overlap_rejected():
    # Net rects are 5 apart, but the facing bleeds overlap.
    payload = {
        "stock": {"width": 200, "height": 100},
        "items": [
            {"id": "A", "width": 40, "height": 40, "quantity": 1,
             "bleed": {"right": 5}},
            {"id": "B", "width": 40, "height": 40, "quantity": 1,
             "bleed": {"left": 5}},
        ],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "B", "x": 45, "y": 0},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "overlap" in response.json()["detail"]


def test_locked_bleed_guillotine_gap_uses_production_rect():
    # A's production rect ends at 45; B starts at 48: kerf 3 fits exactly.
    payload = {
        "stock": {"width": 100, "height": 50},
        "kerf": 3,
        "items": [
            {"id": "A", "width": 40, "height": 50, "quantity": 1,
             "bleed": {"right": 5}},
            {"id": "B", "width": 40, "height": 50, "quantity": 1},
        ],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "B", "x": 48, "y": 0},
        ],
    }
    body = client.post("/plans", json=payload).json()
    check_plan(body)
    # One unit closer and the kerf no longer fits between production rects.
    payload["locked_placements"][1]["x"] = 47
    assert client.post("/plans", json=payload).status_code == 422


def test_gripper_zone_checks_locked_production_rect():
    # Net top edge y=12 clears the inner area start, but the 3mm top
    # bleed reaches into the kerf strip [10, 12).
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1,
                   "bleed": {"top": 3}}],
        "locked_placements": [{"item": "A", "x": 0, "y": 12}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_bleed_item_unfit_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 96, "height": 96, "quantity": 1,
                   "bleed": {"top": 4, "right": 4, "bottom": 4, "left": 4}}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "A" in detail and "bleed" in detail


def test_bleed_rotation_chosen_when_unrotated_production_unfit():
    # Production rect is 70x50 unrotated (too wide for 55) and 50x70
    # rotated, so the engine must rotate.
    payload = {
        "stock": {"width": 55, "height": 80},
        "items": [{"id": "A", "width": 60, "height": 40, "quantity": 1,
                   "bleed": {"top": 5, "right": 5, "bottom": 5, "left": 5}}],
    }
    body = client.post("/plans", json=payload).json()
    check_plan(body)
    p = body["sheets"][0]["placements"][0]
    assert p["rotated"] is True
    assert (p["production_rect"]["width"],
            p["production_rect"]["height"]) == (50.0, 70.0)
