"""Request/response schemas for the cutting plan API."""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# Sanity limits guarding the service against absurd workloads.
MAX_DIMENSION = 100_000.0
MAX_KERF = 1_000.0
MAX_ITEM_TYPES = 500
MAX_QUANTITY = 100_000
MAX_TOTAL_PIECES = 20_000


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------

class Stock(BaseModel):
    width: float = Field(gt=0, le=MAX_DIMENSION, description="原纸宽度")
    height: float = Field(gt=0, le=MAX_DIMENSION, description="原纸高度")


class Bleed(BaseModel):
    """Bleed margins (出血) around an item's net (finished) size.

    All sides default to 0.  The production rectangle used for packing is
    the net size plus the four bleed margins; when the item is rotated
    90° the margins rotate with it.
    """

    top: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION,
                       description="上边出血宽度")
    right: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION,
                         description="右边出血宽度")
    bottom: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION,
                          description="下边出血宽度")
    left: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION,
                        description="左边出血宽度")


class Item(BaseModel):
    id: Optional[str] = Field(
        default=None, min_length=1, max_length=64,
        description="成品标识，缺省时自动生成 item-1、item-2 …",
    )
    width: float = Field(gt=0, le=MAX_DIMENSION, description="成品净宽度")
    height: float = Field(gt=0, le=MAX_DIMENSION, description="成品净高度")
    bleed: Optional[Bleed] = Field(
        default=None,
        description=(
            "四边出血：成品净尺寸之外随成品一起排版的生产留边。排版、旋转与"
            "各种校验均以“净尺寸 + 四边出血”的生产矩形为准；缺省（或 null）"
            "四边均为 0"
        ),
    )
    quantity: int = Field(ge=1, le=MAX_QUANTITY, description="需求数量")


class GripperZone(BaseModel):
    edge: Literal["top", "right", "bottom", "left"] = Field(
        description="叼口所在原纸边：top 上边、right 右边、bottom 下边、left 左边"
    )
    size: float = Field(
        gt=0, le=MAX_DIMENSION,
        description="叼口保留区宽度（垂直于该边的尺寸），整条边区域均不得放置成品",
    )


class LockedPlacement(BaseModel):
    item: str = Field(min_length=1, max_length=64,
                      description="预先定位成品的标识，须出现在 items 中")
    x: float = Field(ge=0, le=MAX_DIMENSION,
                     description="成品净尺寸左上角 X 坐标（原纸左上角为原点）")
    y: float = Field(ge=0, le=MAX_DIMENSION,
                     description="成品净尺寸左上角 Y 坐标")
    rotated: bool = Field(default=False, description="是否旋转了 90 度")


class PlanRequest(BaseModel):
    stock: Stock = Field(description="原纸尺寸")
    kerf: float = Field(default=0.0, ge=0.0, le=MAX_KERF,
                        description="切缝宽度，每次切割损耗的材料宽度")
    max_cuts_per_sheet: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "每张原纸允许的最大刀数（按 cut_tree 中 cut 节点计数），"
            "缺省表示不限制"
        ),
    )
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEM_TYPES,
                              description="成品尺寸及数量")
    locked_placements: list[LockedPlacement] = Field(
        default_factory=list, max_length=MAX_TOTAL_PIECES,
        description=(
            "预先定位、必须原样保留的成品（均位于第一张原纸）。x/y 为成品"
            "净尺寸左上角；若成品设置了 bleed，引擎据此反推出生产矩形"
            "（净尺寸 + 四边出血），并以生产矩形做越界、重叠、叼口与铡刀"
            "可分解校验。锁定数量计入对应 item 的需求数量；布局必须能在"
            "给定 kerf 下递归分解为铡刀切割树。"
        ),
    )
    gripper_zone: Optional[GripperZone] = Field(
        default=None,
        description=(
            "机台叼口保留区：该边整条区域必须先用一道切缝为 kerf 的分割刀"
            "从原纸切出并记为 waste，其内不得放置任何成品；缺省时无保留区。"
        ),
    )

    @model_validator(mode="after")
    def _check_request(self) -> "PlanRequest":
        ids = [item.id for item in self.items if item.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique")
        total = sum(item.quantity for item in self.items)
        if total > MAX_TOTAL_PIECES:
            raise ValueError(
                f"total piece count {total} exceeds the limit of {MAX_TOTAL_PIECES}"
            )
        return self


# ---------------------------------------------------------------------------
# response
# ---------------------------------------------------------------------------

class ProductionRect(BaseModel):
    """The rectangle actually packed and cut: net size plus bleed."""

    x: float = Field(description="生产矩形左上角 X 坐标（原纸左上角为原点）")
    y: float = Field(description="生产矩形左上角 Y 坐标")
    width: float = Field(description="生产矩形宽度（净宽 + 左右出血）")
    height: float = Field(description="生产矩形高度（净高 + 上下出血）")


class Placement(BaseModel):
    item: str = Field(description="成品标识")
    x: float = Field(description="成品净尺寸左上角 X 坐标（原纸左上角为原点）")
    y: float = Field(description="成品净尺寸左上角 Y 坐标")
    width: float = Field(description="净尺寸放置后的宽度（旋转后为原高度）")
    height: float = Field(description="净尺寸放置后的高度")
    rotated: bool = Field(description="是否旋转了 90 度")
    locked: bool = Field(default=False,
                         description="是否为请求中预先定位（locked）的成品")
    production_rect: Optional[ProductionRect] = Field(
        default=None,
        description=(
            "净尺寸加四边出血后的生产矩形；排版与 cut_tree 的 item 叶均覆盖"
            "该矩形。仅当该成品设置了非零 bleed 时返回"
        ),
    )


class ItemNode(BaseModel):
    """A placed piece; the leaf rectangle is its production rectangle
    (net size plus bleed), matching ``production_rect`` in placements."""

    type: Literal["item"] = "item"
    item: str
    x: float
    y: float
    width: float
    height: float
    rotated: bool


class WasteNode(BaseModel):
    type: Literal["waste"] = "waste"
    x: float
    y: float
    width: float
    height: float


class CutNode(BaseModel):
    """An internal node: one guillotine cut spanning the whole rectangle.

    "vertical" cuts produce a left (first) and right (second) child,
    "horizontal" cuts a top (first) and bottom (second) child.  The second
    child starts exactly ``kerf`` units after the first child ends.
    """

    type: Literal["cut"] = "cut"
    direction: Literal["vertical", "horizontal"]
    x: float
    y: float
    width: float
    height: float
    children: list["TreeNode"] = Field(min_length=2, max_length=2)


TreeNode = Annotated[
    Union[CutNode, ItemNode, WasteNode],
    Field(discriminator="type"),
]
CutNode.model_rebuild()


class SheetPlan(BaseModel):
    index: int = Field(description="原纸序号，从 0 开始")
    width: float
    height: float
    utilization: float = Field(ge=0.0, le=1.0,
                               description="成品净面积 / 原纸面积（不含出血）")
    cut_count: int = Field(ge=0, description="切割树中的 cut 节点（刀）数")
    placements: list[Placement]
    cut_tree: TreeNode = Field(description="与 placements 坐标一致的二叉切割树")


class Summary(BaseModel):
    sheet_count: int
    piece_count: int
    used_area: float
    stock_area: float
    utilization: float


class PlanResponse(BaseModel):
    stock: Stock
    kerf: float
    sheets: list[SheetPlan]
    summary: Summary
