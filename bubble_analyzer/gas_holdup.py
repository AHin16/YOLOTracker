#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


class EdgeBubblePolicy(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    WEIGHTED = "weighted"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True)
class ROI:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def resolve(self, frame_shape: Sequence[int]) -> Tuple[int, int, int, int]:
        frame_height, frame_width = _normalize_frame_shape(frame_shape)
        x1 = min(max(int(self.x), 0), frame_width)
        y1 = min(max(int(self.y), 0), frame_height)
        x2 = frame_width if int(self.width) <= 0 else min(x1 + int(self.width), frame_width)
        y2 = frame_height if int(self.height) <= 0 else min(y1 + int(self.height), frame_height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI does not overlap the frame.")
        return x1, y1, x2, y2


@dataclass(frozen=True)
class GasHoldupResult:
    bubble_count: int
    bubble_area: int
    roi_area: int
    bubble_area_ratio: float
    gas_holdup: float
    average_diameter: float
    accepted_masks: Tuple[np.ndarray, ...]
    roi_bounds: Tuple[int, int, int, int]


def _normalize_frame_shape(frame_shape: Sequence[int]) -> Tuple[int, int]:
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width.")
    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("frame dimensions must be greater than zero.")
    return frame_height, frame_width


def _touches_frame_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def analyze_masks(
    masks: Iterable[np.ndarray],
    frame_shape: Sequence[int],
    min_bubble_area: int = 10,
    roi: Optional[ROI] = None,
    edge_policy: EdgeBubblePolicy = EdgeBubblePolicy.INCLUDE,
) -> GasHoldupResult:
    frame_height, frame_width = _normalize_frame_shape(frame_shape)
    resolved_roi = (roi or ROI()).resolve((frame_height, frame_width))
    x1, y1, x2, y2 = resolved_roi
    roi_area = (x2 - x1) * (y2 - y1)
    minimum_area = max(int(min_bubble_area), 0)
    policy = EdgeBubblePolicy.from_value(edge_policy)

    accepted_masks = []
    instance_areas = []
    union_mask = np.zeros((frame_height, frame_width), dtype=bool)

    for source_mask in masks:
        mask = np.asarray(source_mask)
        if mask.shape[:2] != (frame_height, frame_width) or mask.ndim != 2:
            raise ValueError(
                f"mask shape {mask.shape} does not match frame shape "
                f"{(frame_height, frame_width)}."
            )

        visible_mask = mask.astype(bool, copy=False)
        if policy == EdgeBubblePolicy.EXCLUDE and _touches_frame_border(visible_mask):
            continue

        roi_mask = np.zeros_like(visible_mask)
        roi_mask[y1:y2, x1:x2] = visible_mask[y1:y2, x1:x2]
        area = int(np.count_nonzero(roi_mask))
        if area < minimum_area:
            continue

        accepted_masks.append(roi_mask)
        instance_areas.append(area)
        union_mask |= roi_mask

    bubble_area = int(np.count_nonzero(union_mask))
    ratio_percent = 100.0 * bubble_area / roi_area
    diameters = [2.0 * math.sqrt(area / math.pi) for area in instance_areas]
    average_diameter = float(np.mean(diameters)) if diameters else 0.0

    return GasHoldupResult(
        bubble_count=len(accepted_masks),
        bubble_area=bubble_area,
        roi_area=roi_area,
        bubble_area_ratio=ratio_percent,
        gas_holdup=ratio_percent,
        average_diameter=average_diameter,
        accepted_masks=tuple(accepted_masks),
        roi_bounds=resolved_roi,
    )
