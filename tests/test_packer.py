import pytest

from app.packer import (
    CutLimitError,
    GripperZoneError,
    LockedLayoutError,
    Node,
    Piece,
    Sheet,
    UnfitItemError,
    _best_fit,
    build_plan,
)
from tests.check_plan import EPS, check_plan, count_cut_nodes, iter_leaves


def plan(stock_w, stock_h, kerf, items):
    return build_plan(stock_w, stock_h, kerf, items)


class TestBasicPacking:
    def test_exact_fit_single_piece(self):
        result = plan(100, 100, 3, [("A", 100, 100, 1)])
        assert result["summary"]["sheet_count"] == 1
        sheet = result["sheets"][0]
        assert sheet["utilization"] == 1.0
        assert sheet["placements"] == [
            {"item": "A", "x": 0, "y": 0, "width": 100, "height": 100,
             "rotated": False, "locked": False}
        ]
        assert sheet["cut_tree"]["type"] == "item"

    def test_kerf_is_consumed_between_pieces(self):
        # Two 45-wide pieces on a 100-wide sheet with kerf 5:
        # the second must start at 45 + 5 = 50, not 45.
        result = plan(100, 100, 5, [("A", 45, 100, 2)])
        sheet = result["sheets"][0]
        assert result["summary"]["sheet_count"] == 1
        xs = sorted(p["x"] for p in sheet["placements"])
        assert xs == [0, 50]
        root = sheet["cut_tree"]
        assert root["type"] == "cut" and root["direction"] == "vertical"
        first, second = root["children"]
        assert second["x"] == first["x"] + first["width"] + 5

    def test_kerf_can_force_an_extra_sheet(self):
        # 3 pieces of 35 need 35*3 + 2*4 = 113 > 100 -> two sheets.
        result = plan(100, 100, 4, [("A", 35, 100, 3)])
        assert result["summary"]["sheet_count"] == 2
        check_plan(result)

    def test_zero_kerf_packs_tightly(self):
        result = plan(100, 100, 0, [("A", 50, 100, 2)])
        assert result["summary"]["sheet_count"] == 1
        assert result["sheets"][0]["utilization"] == 1.0

    def test_rotation_when_unrotated_does_not_fit(self):
        result = plan(50, 100, 0, [("A", 60, 40, 1)])
        placement = result["sheets"][0]["placements"][0]
        assert placement["rotated"] is True
        assert placement["width"] == 40
        assert placement["height"] == 60

    def test_unrotated_preferred_when_both_fit(self):
        result = plan(100, 100, 0, [("A", 60, 40, 1)])
        placement = result["sheets"][0]["placements"][0]
        assert placement["rotated"] is False
        assert placement["width"] == 60

    def test_multiple_sheets(self):
        # 8 pieces of 50x50 pack exactly 4 per 100x100 sheet.
        result = plan(100, 100, 0, [("A", 50, 50, 8)])
        assert result["summary"]["sheet_count"] == 2
        assert result["summary"]["utilization"] == 1.0
        check_plan(result)

    def test_multiple_sheets_and_items(self):
        result = plan(100, 100, 0, [("A", 60, 60, 3), ("B", 40, 40, 4)])
        assert result["summary"]["sheet_count"] >= 2
        assert result["summary"]["piece_count"] == 7
        check_plan(result)


class TestUnfitItems:
    def test_item_larger_than_stock_raises(self):
        with pytest.raises(UnfitItemError) as excinfo:
            plan(100, 100, 0, [("big", 101, 50, 1)])
        assert "big" in str(excinfo.value)

    def test_item_unfit_in_both_orientations_raises(self):
        with pytest.raises(UnfitItemError):
            plan(100, 60, 0, [("awkward", 120, 70, 1)])

    def test_fit_check_ignores_kerf_on_fresh_sheet(self):
        # A piece matching the stock exactly must fit even with a large kerf:
        # no cut is needed at the sheet edges.
        result = plan(100, 100, 10, [("A", 100, 100, 1)])
        assert result["summary"]["sheet_count"] == 1


class TestTreeConsistency:
    def test_mixed_batch_tree_is_consistent(self):
        result = plan(
            900, 600, 3,
            [("flyer", 210, 148, 8), ("card", 90, 55, 20), ("tag", 120, 80, 5)],
        )
        check_plan(result)

    def test_waste_leaves_have_positive_size(self):
        result = plan(100, 100, 5, [("A", 30, 30, 3)])
        check_plan(result)
        for sheet in result["sheets"]:
            for leaf in iter_leaves(sheet["cut_tree"]):
                if leaf["type"] == "waste":
                    assert leaf["width"] > 0
                    assert leaf["height"] > 0

    def test_trim_sliver_narrower_than_kerf_is_allowed(self):
        # 300-wide piece on 302-wide stock with kerf 3: the 2mm sliver
        # cannot host anything (the blade would eat it) -> trim loss,
        # the piece must still be placed.
        result = plan(302, 100, 3, [("A", 300, 100, 1)])
        assert result["summary"]["sheet_count"] == 1
        check_plan(result)

    def test_decimal_dimensions(self):
        result = plan(297.5, 210.0, 1.5, [("A", 99.1, 70.3, 6)])
        check_plan(result)

    def test_utilization_never_exceeds_one(self):
        result = plan(200, 100, 2, [("A", 45, 45, 30)])
        check_plan(result)
        for sheet in result["sheets"]:
            assert 0 < sheet["utilization"] <= 1.0


# ---------------------------------------------------------------------------
# locked placements
# ---------------------------------------------------------------------------

def _locked(items, pinned, stock_w=100, stock_h=100, kerf=0, max_cuts=None):
    refs = [(iid, x, y, rot, i + 1)
            for i, (iid, x, y, rot) in enumerate(pinned)]
    return build_plan(stock_w, stock_h, kerf, items, locked=refs,
                      max_cuts_per_sheet=max_cuts)


class TestLockedPlacements:
    def test_single_locked_piece_marked_and_counted(self):
        result = _locked([("A", 40, 40, 3)],
                         [("A", 0, 0, False)], kerf=0)
        check_plan(result)
        assert result["summary"]["piece_count"] == 3
        locked = [p for s in result["sheets"] for p in s["placements"]
                  if p["locked"]]
        assert len(locked) == 1
        assert locked[0]["item"] == "A"

    def test_locked_coordinates_are_preserved(self):
        result = _locked([("A", 40, 40, 1)],
                         [("A", 30, 20, False)], stock_w=100, stock_h=100)
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert (p["x"], p["y"], p["width"], p["height"]) == (30, 20, 40, 40)
        assert p["locked"] is True

    def test_rotated_locked_piece(self):
        result = _locked([("A", 60, 40, 1)], [("A", 40, 0, True)])
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert p["rotated"] is True
        assert (p["width"], p["height"]) == (40, 60)

    def test_two_locked_pieces_exact_kerf_gap(self):
        # 45 + kerf 5 + 45 == 95 <= 100: the cut sits exactly between them.
        result = _locked(
            [("A", 45, 40, 3)],
            [("A", 0, 0, False), ("A", 50, 0, False)], kerf=5)
        check_plan(result)
        assert result["summary"]["piece_count"] == 3
        root = result["sheets"][0]["cut_tree"]
        assert root["type"] == "cut" and root["direction"] == "vertical"
        xs = sorted(p["x"] for p in result["sheets"][0]["placements"]
                    if p["locked"])
        assert xs == [0, 50]

    def test_locked_grid_with_exact_kerf(self):
        pinned = [("A", 0, 0, False), ("A", 50, 0, False),
                  ("A", 0, 50, False), ("A", 50, 50, False)]
        result = _locked([("A", 40, 40, 4)], pinned, kerf=10)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 1

    def test_remaining_pieces_fill_usable_offcuts(self):
        # Three locked corners leave the bottom-right 40x40 region free.
        pinned = [("A", 0, 0, False), ("A", 50, 0, False),
                  ("A", 0, 50, False)]
        result = _locked([("A", 40, 40, 4)], pinned, kerf=10)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 1
        assert result["summary"]["piece_count"] == 4
        unlocked = [p for s in result["sheets"] for p in s["placements"]
                    if not p["locked"]]
        assert len(unlocked) == 1
        assert (unlocked[0]["x"], unlocked[0]["y"]) == (50, 50)

    def test_remaining_pieces_overflow_to_new_sheets(self):
        pinned = [("A", 0, 0, False), ("A", 50, 0, False)]
        result = _locked([("A", 45, 100, 4)], pinned,
                         stock_w=100, stock_h=100, kerf=5)
        check_plan(result)
        assert result["summary"]["sheet_count"] >= 2
        assert result["summary"]["piece_count"] == 4
        # Locked pieces stay on sheet 0.
        assert all(p["locked"] for p in result["sheets"][0]["placements"])

    def test_interior_piece_peels_free_strips(self):
        result = _locked([("A", 100, 80, 1)],
                         [("A", 150, 100, False)],
                         stock_w=300, stock_h=200, kerf=3)
        check_plan(result)
        leaves = list(iter_leaves(result["sheets"][0]["cut_tree"]))
        assert any(leaf["type"] == "item" for leaf in leaves)
        waste = [leaf for leaf in leaves if leaf["type"] == "waste"]
        assert waste  # peeled strips end as waste

    def test_locked_strip_is_reused_when_piece_fits_cleanly(self):
        # 20-unit left gap with kerf 3 yields a 17-wide free strip.
        result = _locked([("A", 40, 40, 1), ("B", 17, 100, 1)],
                         [("A", 20, 20, False)], kerf=3)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 1

    def test_sub_kerf_sliver_blocks_reuse(self):
        # A 15-wide piece leaves 2 units in the 17-wide strip: narrower than
        # the kerf, so the piece must go on a new sheet instead.
        result = _locked([("A", 40, 40, 1), ("B", 15, 100, 1)],
                         [("A", 20, 20, False)], kerf=3)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 2

    def test_touching_pieces_with_kerf_are_rejected(self):
        with pytest.raises(LockedLayoutError) as exc:
            _locked([("A", 45, 40, 2)],
                    [("A", 0, 0, False), ("A", 45, 0, False)], kerf=3)
        message = str(exc.value)
        assert "#1" in message and "#2" in message

    def test_gap_smaller_than_kerf_is_rejected(self):
        with pytest.raises(LockedLayoutError):
            _locked([("A", 45, 40, 2)],
                    [("A", 0, 0, False), ("A", 48, 0, False)], kerf=5)

    def test_origin_clearance_smaller_than_kerf_rejected(self):
        with pytest.raises(LockedLayoutError) as exc:
            _locked([("A", 40, 40, 1)], [("A", 2, 0, False)], kerf=5)
        assert "#1" in str(exc.value)

    def test_non_guillotine_windmill_rejected(self):
        # Classic non-slicing layout: four interlocking blades.
        coords = [(0, 0, 80, 120), (80, 0, 120, 80),
                  (160, 80, 40, 120), (0, 120, 160, 80)]
        items = [(f"p{i}", w, h, 1) for i, (x, y, w, h) in enumerate(coords, 1)]
        pinned = [(f"p{i}", x, y, False)
                  for i, (x, y, w, h) in enumerate(coords, 1)]
        with pytest.raises(LockedLayoutError) as exc:
            _locked(items, pinned, stock_w=200, stock_h=200, kerf=0)
        # Conflict report names the stuck placements.
        assert "p1" in str(exc.value)

    def test_overlapping_locked_pieces_rejected(self):
        with pytest.raises(LockedLayoutError) as exc:
            _locked([("A", 40, 40, 2)],
                    [("A", 0, 0, False), ("A", 30, 30, False)])
        assert "#1" in str(exc.value) and "#2" in str(exc.value)

    def test_out_of_bounds_locked_piece_rejected(self):
        with pytest.raises(LockedLayoutError):
            _locked([("A", 40, 40, 1)], [("A", 70, 70, False)])

    def test_rotated_piece_out_of_bounds_rejected(self):
        with pytest.raises(LockedLayoutError):
            _locked([("A", 60, 40, 1)], [("A", 50, 0, True)],
                    stock_w=100, stock_h=50)

    def test_summary_covers_locked_and_remaining(self):
        pinned = [("flyer", 0, 0, False), ("card", 0, 151, False)]
        result = _locked(
            [("flyer", 210, 148, 8), ("card", 90, 55, 20)], pinned,
            stock_w=900, stock_h=600, kerf=3)
        check_plan(result)
        assert result["summary"]["piece_count"] == 28
        used = sum(p["width"] * p["height"]
                   for s in result["sheets"] for p in s["placements"])
        assert abs(result["summary"]["used_area"] - round(used, 6)) <= 1e-4

    def test_large_locked_grid_decomposes(self):
        n = 12
        pinned = [("A", x * 11, y * 11, False)
                  for y in range(n) for x in range(n)]
        result = _locked([("A", 10, 10, n * n)], pinned,
                         stock_w=n * 11, stock_h=n * 11, kerf=1)
        check_plan(result)
        assert result["summary"]["piece_count"] == n * n


# ---------------------------------------------------------------------------
# per-sheet cut limit
# ---------------------------------------------------------------------------

class TestCutLimit:
    def test_cut_count_reported_without_limit(self):
        result = plan(100, 100, 5, [("A", 45, 100, 2)])
        assert result["sheets"][0]["cut_count"] == 1

    def test_exact_fit_reports_zero_cuts(self):
        result = plan(100, 100, 3, [("A", 100, 100, 1)])
        assert result["sheets"][0]["cut_count"] == 0

    def test_cut_count_matches_cut_tree(self):
        result = plan(900, 600, 3,
                      [("flyer", 210, 148, 8), ("card", 90, 55, 20)])
        check_plan(result)
        for sheet in result["sheets"]:
            assert sheet["cut_count"] == count_cut_nodes(sheet["cut_tree"])

    def test_limit_splits_across_sheets(self):
        # A 50x100 piece on a 100-wide sheet costs one cut; the second
        # fills the offcut exactly (no cut).  With a limit of 1 only two
        # pieces share a sheet.
        result = build_plan(100, 100, 0, [("A", 50, 100, 4)],
                            max_cuts_per_sheet=1)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 2
        assert all(s["cut_count"] <= 1 for s in result["sheets"])

    def test_item_too_demanding_for_limit_rejected(self):
        # A 40x40 piece on a blank 100x100 sheet (kerf 3) needs two cuts.
        with pytest.raises(CutLimitError) as excinfo:
            build_plan(100, 100, 3, [("A", 40, 40, 1)], max_cuts_per_sheet=1)
        assert "A" in str(excinfo.value)

    def test_item_at_exact_limit_accepted(self):
        result = build_plan(100, 100, 3, [("A", 40, 40, 1)],
                            max_cuts_per_sheet=2)
        check_plan(result)
        assert result["sheets"][0]["cut_count"] == 2

    def test_locked_tree_over_limit_rejected(self):
        # The locked pair needs three cuts (one split + one per piece).
        pinned = [("A", 0, 0, False), ("A", 50, 0, False)]
        with pytest.raises(CutLimitError):
            _locked([("A", 45, 40, 2)], pinned, kerf=5, max_cuts=2)

    def test_locked_tree_at_limit_accepted(self):
        pinned = [("A", 0, 0, False), ("A", 50, 0, False)]
        result = _locked([("A", 45, 40, 2)], pinned, kerf=5, max_cuts=3)
        check_plan(result)
        assert result["sheets"][0]["cut_count"] == 3

    def test_locked_cuts_consume_sheet_budget(self):
        # The locked tree uses all 3 cuts, so the remaining piece cannot
        # split any offcut of the first sheet and overflows to a new one.
        pinned = [("A", 0, 0, False), ("A", 50, 0, False)]
        result = _locked([("A", 45, 40, 3)], pinned, kerf=5, max_cuts=3)
        check_plan(result)
        assert result["summary"]["sheet_count"] == 2
        assert result["sheets"][0]["cut_count"] == 3
        assert result["sheets"][1]["cut_count"] == 2


def _sheet_with_free(rects, kerf, cuts=0):
    sheet = Sheet(width=100.0, height=100.0, kerf=kerf,
                  root=Node(kind="free", x=0.0, y=0.0,
                            width=100.0, height=100.0))
    sheet.free = [Node(kind="free", x=0.0, y=float(10 * i),
                       width=w, height=h)
                  for i, (w, h) in enumerate(rects)]
    sheet.cuts = cuts
    return sheet


class TestBestFitCutLimit:
    def test_equal_fit_prefers_fewer_added_cuts(self):
        # Both rectangles score the same for a 20x10 piece (short side 0,
        # leftover area 60) but the split costs 1 cut vs 0.
        sheet = _sheet_with_free([(26, 10), (20, 13)], kerf=3)
        found = _best_fit([sheet], Piece("A", 20, 10), max_cuts=10)
        assert found is not None
        assert found[1] == 1

    def test_equal_fit_without_limit_keeps_legacy_order(self):
        sheet = _sheet_with_free([(26, 10), (20, 13)], kerf=3)
        found = _best_fit([sheet], Piece("A", 20, 10))
        assert found is not None
        assert found[1] == 0

    def test_rects_exceeding_the_limit_are_skipped(self):
        # The split adds one cut; the sheet is already at the limit.
        sheet = _sheet_with_free([(26, 10)], kerf=3, cuts=1)
        piece = Piece("A", 20, 10)
        assert _best_fit([sheet], piece, max_cuts=1) is None
        assert _best_fit([sheet], piece, max_cuts=2) is not None


# ---------------------------------------------------------------------------
# gripper zone (机台叼口保留区)
# ---------------------------------------------------------------------------

GRIPPER_CASES = {
    # edge: (separator direction, waste strip rect, exact-fill piece size)
    "top": ("horizontal", (0.0, 0.0, 100.0, 10.0), (100.0, 88.0)),
    "bottom": ("horizontal", (0.0, 90.0, 100.0, 10.0), (100.0, 88.0)),
    "left": ("vertical", (0.0, 0.0, 10.0, 100.0), (88.0, 100.0)),
    "right": ("vertical", (90.0, 0.0, 10.0, 100.0), (88.0, 100.0)),
}


class TestGripperZone:
    @pytest.mark.parametrize("edge", ["top", "right", "bottom", "left"])
    def test_strip_cut_off_as_waste_on_every_sheet(self, edge):
        direction, waste_rect, _ = GRIPPER_CASES[edge]
        result = build_plan(100, 100, 2, [("A", 40, 40, 6)],
                            gripper=(edge, 10))
        check_plan(result)
        for sheet in result["sheets"]:
            root = sheet["cut_tree"]
            assert root["type"] == "cut"
            assert root["direction"] == direction
            leaves = list(iter_leaves(root))
            strips = [l for l in leaves
                      if (l["type"], l["x"], l["y"],
                          l["width"], l["height"]) == ("waste",) + waste_rect]
            assert len(strips) == 1

    @pytest.mark.parametrize("edge", ["top", "right", "bottom", "left"])
    def test_no_piece_enters_the_zone(self, edge):
        result = build_plan(100, 100, 2, [("A", 40, 40, 8)],
                            gripper=(edge, 10))
        check_plan(result)
        for sheet in result["sheets"]:
            for p in sheet["placements"]:
                if edge == "top":
                    assert p["y"] >= 12 - EPS
                elif edge == "bottom":
                    assert p["y"] + p["height"] <= 88 + EPS
                elif edge == "left":
                    assert p["x"] >= 12 - EPS
                else:
                    assert p["x"] + p["width"] <= 88 + EPS

    @pytest.mark.parametrize("edge", ["top", "right", "bottom", "left"])
    def test_separator_cut_is_counted(self, edge):
        # A single piece fills the inner rectangle exactly: the only cut
        # is the gripper separator.
        direction, waste_rect, (pw, ph) = GRIPPER_CASES[edge]
        result = build_plan(100, 100, 2, [("A", pw, ph, 1)],
                            gripper=(edge, 10))
        check_plan(result)
        sheet = result["sheets"][0]
        assert sheet["cut_count"] == 1
        leaves = list(iter_leaves(sheet["cut_tree"]))
        assert len(leaves) == 2
        assert {l["type"] for l in leaves} == {"item", "waste"}

    def test_kerf_zero_separator_is_knife_edge(self):
        result = build_plan(100, 100, 0, [("A", 100, 90, 1)],
                            gripper=("top", 10))
        check_plan(result)
        assert result["sheets"][0]["cut_count"] == 1
        p = result["sheets"][0]["placements"][0]
        assert (p["y"], p["height"]) == (10, 90)

    def test_strip_reduces_capacity(self):
        # Inner height is 88 after the strip and kerf: eight 10-high rows
        # still fit per sheet either way, but each additional overflow
        # sheet loses the same capacity too, so filling exactly 16 rows
        # needs 2 sheets without a zone and 3 with it.
        without = build_plan(100, 100, 2, [("A", 100, 10, 16)])
        with_zone = build_plan(100, 100, 2, [("A", 100, 10, 16)],
                               gripper=("top", 10))
        check_plan(with_zone)
        assert without["summary"]["sheet_count"] == 2
        assert with_zone["summary"]["sheet_count"] == 3

    def test_no_gripper_keeps_legacy_plan(self):
        args = (100, 100, 3, [("A", 30, 30, 3), ("B", 25, 20, 5)])
        assert build_plan(*args) == build_plan(*args, gripper=None)

    def test_inner_area_without_positive_size_rejected(self):
        with pytest.raises(GripperZoneError):
            build_plan(100, 100, 2, [("A", 10, 10, 1)],
                       gripper=("top", 100))

    def test_item_fitting_stock_but_not_inner_rejected(self):
        with pytest.raises(GripperZoneError) as exc:
            build_plan(100, 100, 2, [("A", 95, 95, 1)],
                       gripper=("top", 10))
        assert "A" in str(exc.value)
        assert "gripper_zone" in str(exc.value)

    def test_separator_cut_counts_towards_limit(self):
        # 40x40 inside 100x88 with kerf 2 costs two more cuts after the
        # separator: total 3.
        with pytest.raises(CutLimitError):
            build_plan(100, 100, 2, [("A", 40, 40, 1)],
                       gripper=("top", 10), max_cuts_per_sheet=2)
        result = build_plan(100, 100, 2, [("A", 40, 40, 1)],
                            gripper=("top", 10), max_cuts_per_sheet=3)
        check_plan(result)
        assert result["sheets"][0]["cut_count"] == 3

    def test_locked_piece_inside_inner_area_accepted(self):
        result = build_plan(
            100, 100, 2, [("A", 40, 40, 2)],
            locked=[("A", 12, 12, False, 1)], gripper=("top", 10))
        check_plan(result)
        assert result["sheets"][0]["placements"][0]["locked"] is True

    def test_locked_piece_entering_zone_rejected(self):
        with pytest.raises(GripperZoneError) as exc:
            build_plan(100, 100, 2, [("A", 40, 40, 1)],
                       locked=[("A", 0, 5, False, 1)],
                       gripper=("top", 10))
        message = str(exc.value)
        assert "#1" in message and "A" in message

    def test_locked_piece_straddling_separator_rejected(self):
        # Piece starts at y=11 inside the kerf strip [10, 12).
        with pytest.raises(GripperZoneError):
            build_plan(100, 100, 2, [("A", 40, 40, 1)],
                       locked=[("A", 0, 11, False, 1)],
                       gripper=("top", 10))

    @pytest.mark.parametrize("edge,locked_xy", [
        ("right", (49, 0)),    # ends at 89, inside kerf [88, 90)
        ("bottom", (0, 49)),
        ("left", (11, 0)),
    ])
    def test_locked_straddle_rejected_other_edges(self, edge, locked_xy):
        x, y = locked_xy
        with pytest.raises(GripperZoneError):
            build_plan(100, 100, 2, [("A", 40, 40, 1)],
                       locked=[("A", x, y, False, 1)],
                       gripper=(edge, 10))

    def test_locked_tree_plus_separator_counts_towards_limit(self):
        pinned = [("A", 0, 15, False, 1), ("A", 50, 15, False, 2)]
        with pytest.raises(CutLimitError):
            build_plan(100, 100, 5, [("A", 45, 40, 2)],
                       locked=pinned, gripper=("top", 10),
                       max_cuts_per_sheet=3)
        result = build_plan(100, 100, 5, [("A", 45, 40, 2)],
                            locked=pinned, gripper=("top", 10),
                            max_cuts_per_sheet=4)
        check_plan(result)
        assert result["sheets"][0]["cut_count"] == 4

    def test_error_names_ref_and_item(self):
        with pytest.raises(GripperZoneError) as exc:
            build_plan(100, 100, 2, [("box", 40, 40, 1)],
                       locked=[("box", 5, 0, False, 3)],
                       gripper=("left", 10))
        message = str(exc.value)
        assert "#3" in message and "box" in message


# ---------------------------------------------------------------------------
# bleed (四边出血)
# ---------------------------------------------------------------------------

class TestBleed:
    def test_packing_uses_production_rect(self):
        result = build_plan(100, 100, 0, [("A", 40, 40, 1)],
                            bleeds={"A": (5, 5, 5, 5)})
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        # Net rect is inset by the bleed from the production rect.
        assert (p["x"], p["y"], p["width"], p["height"]) == (5.0, 5.0, 40, 40)
        assert p["production_rect"] == {"x": 0.0, "y": 0.0,
                                        "width": 50.0, "height": 50.0}

    def test_no_bleed_has_no_production_rect(self):
        result = plan(100, 100, 0, [("A", 40, 40, 1)])
        assert "production_rect" not in result["sheets"][0]["placements"][0]

    def test_zero_bleed_has_no_production_rect(self):
        result = build_plan(100, 100, 0, [("A", 40, 40, 1)],
                            bleeds={"A": (0, 0, 0, 0)})
        assert "production_rect" not in result["sheets"][0]["placements"][0]

    def test_no_bleed_keeps_legacy_plan(self):
        args = (100, 100, 3, [("A", 30, 30, 3), ("B", 25, 20, 5)])
        assert build_plan(*args) == build_plan(*args, bleeds=None)

    def test_bleed_reduces_capacity(self):
        # Net 45x45 packs 2x2 on 100x100; the 55x55 production rect does
        # not, so each piece needs its own sheet.
        result = build_plan(100, 100, 0, [("A", 45, 45, 4)],
                            bleeds={"A": (5, 5, 5, 5)})
        check_plan(result)
        assert result["summary"]["sheet_count"] == 4

    def test_utilization_counts_net_area_only(self):
        result = build_plan(100, 100, 0, [("A", 40, 40, 1)],
                            bleeds={"A": (5, 5, 5, 5)})
        assert result["sheets"][0]["utilization"] == 0.16
        assert result["summary"]["used_area"] == 1600.0

    def test_cut_tree_item_leaf_covers_production_rect(self):
        result = build_plan(100, 100, 3, [("A", 40, 40, 2)],
                            bleeds={"A": (5, 5, 5, 5)})
        check_plan(result)
        leaves = [leaf for sheet in result["sheets"]
                  for leaf in iter_leaves(sheet["cut_tree"])
                  if leaf["type"] == "item"]
        assert len(leaves) == 2
        for leaf in leaves:
            assert (leaf["width"], leaf["height"]) == (50.0, 50.0)

    def test_rotation_rotates_asymmetric_bleed(self):
        # Locked and rotated 90° clockwise: the sheet-frame bleed is
        # top=left(4), right=top(1), bottom=right(2), left=bottom(3).
        result = build_plan(300, 300, 0, [("A", 100, 50, 1)],
                            locked=[("A", 10, 20, True, 1)],
                            bleeds={"A": (1, 2, 3, 4)})
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert p["production_rect"] == {"x": 7.0, "y": 16.0,
                                        "width": 54.0, "height": 106.0}

    def test_locked_net_coordinates_preserved_with_bleed(self):
        result = build_plan(300, 300, 0, [("A", 100, 50, 1)],
                            locked=[("A", 10, 20, True, 1)],
                            bleeds={"A": (1, 2, 3, 4)})
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert (p["x"], p["y"], p["width"], p["height"]) == (
            10.0, 20.0, 50.0, 100.0)
        assert p["locked"] is True

    def test_locked_unrotated_production_rect(self):
        result = build_plan(300, 300, 0, [("A", 100, 50, 1)],
                            locked=[("A", 10, 20, False, 1)],
                            bleeds={"A": (1, 2, 3, 4)})
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert p["production_rect"] == {"x": 6.0, "y": 19.0,
                                        "width": 106.0, "height": 54.0}

    def test_locked_production_rect_out_of_bounds_rejected(self):
        # Net rect sits at the origin, but the left bleed crosses the edge.
        with pytest.raises(LockedLayoutError) as exc:
            build_plan(100, 100, 0, [("A", 40, 40, 1)],
                       locked=[("A", 0, 0, False, 1)],
                       bleeds={"A": (0, 0, 0, 5)})
        assert "#1" in str(exc.value)

    def test_locked_bleed_overlap_rejected(self):
        # Net rects are 5 apart, but the facing bleeds overlap.
        with pytest.raises(LockedLayoutError) as exc:
            build_plan(200, 100, 0,
                       [("A", 40, 40, 1), ("B", 40, 40, 1)],
                       locked=[("A", 0, 0, False, 1), ("B", 45, 0, False, 2)],
                       bleeds={"A": (0, 5, 0, 0), "B": (0, 0, 0, 5)})
        assert "overlap" in str(exc.value)

    def test_locked_decomposition_uses_production_rects(self):
        # A's production rect ends at 45; B starts at 48: the kerf strip
        # fits exactly between the production rects.
        result = build_plan(100, 50, 3,
                            [("A", 40, 50, 1), ("B", 40, 50, 1)],
                            locked=[("A", 0, 0, False, 1),
                                    ("B", 48, 0, False, 2)],
                            bleeds={"A": (0, 5, 0, 0)})
        check_plan(result)
        # One unit closer and the kerf no longer fits between the
        # production rects (nets would still allow it).
        with pytest.raises(LockedLayoutError):
            build_plan(100, 50, 3,
                       [("A", 40, 50, 1), ("B", 40, 50, 1)],
                       locked=[("A", 0, 0, False, 1), ("B", 47, 0, False, 2)],
                       bleeds={"A": (0, 5, 0, 0)})

    def test_unfit_item_error_mentions_bleed(self):
        with pytest.raises(UnfitItemError) as exc:
            build_plan(100, 100, 0, [("A", 96, 96, 1)],
                       bleeds={"A": (4, 4, 4, 4)})
        message = str(exc.value)
        assert "A" in message and "bleed" in message

    def test_gripper_fit_check_uses_production_rect(self):
        # 80x80 net fits the 100x88 inner area; the 96x96 production rect
        # does not.
        with pytest.raises(GripperZoneError) as exc:
            build_plan(100, 100, 2, [("A", 80, 80, 1)],
                       gripper=("top", 10), bleeds={"A": (8, 8, 8, 8)})
        assert "gripper_zone" in str(exc.value)

    def test_locked_gripper_check_uses_production_rect(self):
        # Net top edge y=12 clears the inner area, but the 3mm top bleed
        # reaches into the kerf strip [10, 12).
        with pytest.raises(GripperZoneError):
            build_plan(100, 100, 2, [("A", 40, 40, 1)],
                       locked=[("A", 0, 12, False, 1)],
                       gripper=("top", 10), bleeds={"A": (3, 0, 0, 0)})

    def test_auto_rotation_when_unrotated_production_unfit(self):
        # Production rect is 70x50 unrotated (too wide) and 50x70 rotated.
        result = build_plan(55, 80, 0, [("A", 60, 40, 1)],
                            bleeds={"A": (5, 5, 5, 5)})
        check_plan(result)
        p = result["sheets"][0]["placements"][0]
        assert p["rotated"] is True
        assert (p["production_rect"]["width"],
                p["production_rect"]["height"]) == (50.0, 70.0)

    def test_mixed_bleed_and_plain_items(self):
        result = build_plan(900, 600, 3,
                            [("flyer", 210, 148, 8), ("card", 90, 55, 20)],
                            bleeds={"flyer": (3, 3, 3, 3)})
        check_plan(result)
        for sheet in result["sheets"]:
            for p in sheet["placements"]:
                if p["item"] == "flyer":
                    assert "production_rect" in p
                else:
                    assert "production_rect" not in p

    def test_locked_and_remaining_with_bleed(self):
        result = build_plan(200, 100, 3, [("A", 40, 40, 4)],
                            locked=[("A", 10, 10, False, 1)],
                            bleeds={"A": (2, 3, 2, 3)})
        check_plan(result)
        locked = [p for p in result["sheets"][0]["placements"] if p["locked"]]
        assert locked[0]["production_rect"] == {"x": 7.0, "y": 8.0,
                                                "width": 46.0, "height": 44.0}
        assert result["summary"]["piece_count"] == 4
