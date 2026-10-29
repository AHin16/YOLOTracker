from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FrameMeasurement:
    frame_id: int
    timestamp: object
    valid: bool = True
    reason: str = ""
    bubble_count: object = None
    area_ratio: object = None
    complete_count: int = 0
    diameters: list = field(default_factory=list)
    velocities: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    source: str = ""


class FrameMeasurer:
    def __init__(self, roi, scale=0.01, min_area=10, max_distance=40):
        self.roi = roi
        self.scale = scale
        self.min_area = min_area
        self.max_distance = max_distance
        self.previous = []
        self.previous_frame = None
        self.next_id = 1

    def measure(self, masks, shape, frame_id, timestamp, error=""):
        result = FrameMeasurement(frame_id, timestamp, valid=not bool(error), reason=error)
        if error:
            self.previous = []
            self.previous_frame = frame_id
            return result
        height, width = shape[:2]
        x1, y1, x2, y2 = self.roi.resolve(shape)
        union = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        complete = []
        for source_mask in masks:
            mask = np.asarray(source_mask, dtype=bool)
            if mask.ndim != 2 or mask.shape != (height, width):
                return self.measure([], shape, frame_id, timestamp, error="invalid_mask_shape")
            clipped = mask[y1:y2, x1:x2]
            area = int(clipped.sum())
            if area < max(1, self.min_area):
                continue
            union |= clipped
            coordinates_y, coordinates_x = np.nonzero(clipped)
            center = (float(coordinates_x.mean() + x1), float(coordinates_y.mean() + y1))
            touches_edge = bool(
                clipped[0].any() or clipped[-1].any()
                or clipped[:, 0].any() or clipped[:, -1].any()
            )
            sample = {
                "frame": frame_id, "time": timestamp, "track_id": None,
                "x": center[0], "y": center[1], "area_px": area,
                "diameter": None, "rise_velocity": None, "vx": None,
                "turn_angle": None, "reason": "roi_or_image_edge" if touches_edge else "",
            }
            result.samples.append(sample)
            if not touches_edge:
                sample["diameter"] = 2 * math.sqrt(area / math.pi) * self.scale
                result.diameters.append(sample["diameter"])
                complete.append(sample)
        result.bubble_count = len(result.samples)
        result.area_ratio = float(union.sum()) / union.size * 100
        result.complete_count = len(complete)
        has_time = timestamp is not None and math.isfinite(timestamp)
        gap = self.previous_frame is not None and frame_id != self.previous_frame + 1
        matches = self._associate(complete) if has_time and not gap else {}
        for sample_index, sample in enumerate(complete):
            match = matches.get(sample_index)
            if not has_time:
                sample["reason"] = "missing_time"
            elif gap:
                sample["reason"] = "frame_gap"
            elif isinstance(match, str):
                sample["reason"] = match
            elif match is None:
                sample["reason"] = "new_track" if not self.previous else "unmatched_or_distance"
            else:
                elapsed = timestamp - match["time"]
                if elapsed <= 0:
                    sample["reason"] = "invalid_time"
                else:
                    sample["track_id"] = match["track_id"]
                    delta = np.array([sample["x"] - match["x"], sample["y"] - match["y"]])
                    sample["vx"] = float(delta[0] * self.scale / elapsed)
                    sample["rise_velocity"] = float(-delta[1] * self.scale / elapsed)
                    previous_delta = match.get("delta")
                    if previous_delta is not None:
                        denominator = np.linalg.norm(previous_delta) * np.linalg.norm(delta)
                        if denominator > 1e-8:
                            cosine = np.clip(np.dot(previous_delta, delta) / denominator, -1, 1)
                            sample["turn_angle"] = float(np.degrees(np.arccos(cosine)))
                    sample["delta"] = delta
                    result.velocities.append(sample["rise_velocity"])
            if sample["track_id"] is None:
                sample["track_id"] = self.next_id
                self.next_id += 1
        self.previous = complete if has_time else []
        self.previous_frame = frame_id
        return result

    def _associate(self, current):
        if not current or not self.previous:
            return {}
        distances = np.array([
            [math.hypot(item["x"] - prior["x"], item["y"] - prior["y"]) for prior in self.previous]
            for item in current
        ])
        plausible = np.array([
            [0.5 <= item["area_px"] / prior["area_px"] <= 2 for prior in self.previous]
            for item in current
        ]) & (distances <= self.max_distance)
        costs = np.where(plausible, distances, 1e9)
        rows, columns = linear_sum_assignment(costs)
        matches = {}
        for row, column in zip(rows, columns):
            if not plausible[row, column]:
                continue
            distance = distances[row, column]
            row_alternatives = np.delete(costs[row], column)
            column_alternatives = np.delete(costs[:, column], row)
            margin = max(1.0, distance * 0.25)
            if np.any(row_alternatives <= distance + margin) or np.any(column_alternatives <= distance + margin):
                matches[int(row)] = "ambiguous_match"
            else:
                matches[int(row)] = self.previous[int(column)]
        return matches
