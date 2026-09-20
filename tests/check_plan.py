"""Shared validators asserting a plan is physically and structurally sound."""
from __future__ import annotations

EPS = 1e-6


def iter_leaves(node: dict):
    if node["type"] in ("item", "waste"):
        yield node
    else:
        yield from iter_leaves(node["children"][0])
        yield from iter_leaves(node["children"][1])


def count_cut_nodes(node: dict) -> int:
    """Number of cut nodes in the subtree (one guillotine cut each)."""
    if node["type"] != "cut":
        return 0
    return 1 + sum(count_cut_nodes(child) for child in node["children"])


def check_cut_tree(node: dict, kerf: float) -> None:
    """Recursively verify the binary guillotine cut tree.

    Rules enforced:
    * a cut node has exactly two children and spans its whole rectangle;
    * the first child is flush with the node's origin;
    * the second child starts exactly ``kerf`` after the first child ends
      (the cut consumes the kerf);
    * children never exceed the node's bounds (edge trim is allowed, so the
      far edge of the second child may fall short of the node's far edge);
    * leaves have positive dimensions.
    """
    if node["type"] != "cut":
        assert node["width"] > 0 and node["height"] > 0
        return

    assert node["direction"] in ("vertical", "horizontal")
    assert len(node["children"]) == 2
    first, second = node["children"]

    if node["direction"] == "vertical":
        # Cut line is vertical and spans the full height of the node.
        assert abs(first["x"] - node["x"]) <= EPS
        assert abs(first["y"] - node["y"]) <= EPS
        assert abs(second["y"] - node["y"]) <= EPS
        assert abs(second["x"] - (first["x"] + first["width"] + kerf)) <= EPS
        assert first["height"] <= node["height"] + EPS
        assert second["height"] <= node["height"] + EPS
        assert second["x"] + second["width"] <= node["x"] + node["width"] + EPS
    else:
        # Cut line is horizontal and spans the full width of the node.
        assert abs(first["x"] - node["x"]) <= EPS
        assert abs(first["y"] - node["y"]) <= EPS
        assert abs(second["x"] - node["x"]) <= EPS
        assert abs(second["y"] - (first["y"] + first["height"] + kerf)) <= EPS
        assert first["width"] <= node["width"] + EPS
        assert second["width"] <= node["width"] + EPS
        assert second["y"] + second["height"] <= node["y"] + node["height"] + EPS

    check_cut_tree(first, kerf)
    check_cut_tree(second, kerf)


def check_sheet(sheet: dict, kerf: float) -> None:
    """Verify one sheet: tree structure, placements match tree leaves,
    production rects are inside the sheet and do not overlap, the net
    rect sits inside its production rect, utilization counts net area."""
    width, height = sheet["width"], sheet["height"]
    placements = sheet["placements"]

    check_cut_tree(sheet["cut_tree"], kerf)

    # The reported cut count must match the actual cut tree.
    assert sheet["cut_count"] == count_cut_nodes(sheet["cut_tree"])

    # Tree item leaves cover the placements' production rectangles (the
    # net rect itself when no bleed is reported).
    tree_items = [
        (leaf["item"], leaf["x"], leaf["y"], leaf["width"], leaf["height"],
         leaf["rotated"])
        for leaf in iter_leaves(sheet["cut_tree"])
        if leaf["type"] == "item"
    ]
    placed = []
    for p in placements:
        pr = p.get("production_rect") or p
        placed.append((p["item"], pr["x"], pr["y"], pr["width"],
                       pr["height"], p["rotated"]))
    assert sorted(map(repr, tree_items)) == sorted(map(repr, placed))

    # Bounds and pairwise overlap of the production rectangles; the net
    # rectangle must sit inside its production rectangle.
    rects = []
    for p in placements:
        pr = p.get("production_rect") or p
        assert pr["x"] <= p["x"] + EPS
        assert pr["y"] <= p["y"] + EPS
        assert pr["x"] + pr["width"] >= p["x"] + p["width"] - EPS
        assert pr["y"] + pr["height"] >= p["y"] + p["height"] - EPS
        x2, y2 = pr["x"] + pr["width"], pr["y"] + pr["height"]
        assert pr["x"] >= -EPS and pr["y"] >= -EPS
        assert x2 <= width + EPS and y2 <= height + EPS
        rects.append((pr["x"], pr["y"], x2, y2))
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= EPS or overlap_y <= EPS, (
                f"placements {i} and {j} overlap"
            )

    # Utilization matches the placed net (finished-size) area.
    used = sum(p["width"] * p["height"] for p in placements)
    assert abs(sheet["utilization"] - used / (width * height)) <= 1e-4


def check_plan(plan: dict) -> None:
    kerf = plan["kerf"]
    for sheet in plan["sheets"]:
        assert sheet["width"] == plan["stock"]["width"]
        assert sheet["height"] == plan["stock"]["height"]
        check_sheet(sheet, kerf)

    summary = plan["summary"]
    assert summary["sheet_count"] == len(plan["sheets"])
    assert summary["piece_count"] == sum(
        len(sheet["placements"]) for sheet in plan["sheets"]
    )
