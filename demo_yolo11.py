#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
import os
import os.path as osp
import re
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from bubble_analyzer.image_io import IMAGE_EXT, get_image_list, read_image
from bubble_analyzer.inference import create_predictor

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

from tracking.bot_sort import BOTSORT
from tracking.byte_tracker import BYTETracker
from track_utils.my_timer import MyTimer
from visualization.visualize import plot_tracking


ROOT_DIR = osp.abspath(osp.dirname(__file__))
DESKTOP_BUBBLE_VIDEO_PATH = r"C:\Users\32956\Desktop\2026_04_13_18_42_05\png.mp4"
SAMPLE_VIDEO_PATH = osp.join(ROOT_DIR, "data", "videos", "palace.mp4")
DEFAULT_IMAGE_DIR = osp.join(ROOT_DIR, "data", "dataset", "test_images")
DEFAULT_VIDEO_PATH = DESKTOP_BUBBLE_VIDEO_PATH if osp.isfile(DESKTOP_BUBBLE_VIDEO_PATH) else SAMPLE_VIDEO_PATH
DEFAULT_OUTPUT_DIR = (
    osp.dirname(DEFAULT_VIDEO_PATH)
    if osp.isfile(DEFAULT_VIDEO_PATH) and osp.dirname(DEFAULT_VIDEO_PATH)
    else osp.join(ROOT_DIR, "visualization", "outputs")
)
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def ensure_parent_dir(file_path):
    parent = osp.dirname(osp.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def prepare_output(args, current_time, output_name, frame_size):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    explicit_video_path = getattr(args, "output_video_path", "") or ""

    if explicit_video_path:
        save_path = osp.abspath(explicit_video_path)
        save_folder = osp.dirname(save_path) or osp.abspath(getattr(args, "save_result", DEFAULT_OUTPUT_DIR))
        os.makedirs(save_folder, exist_ok=True)
    else:
        base_output_dir = osp.abspath(getattr(args, "save_result", DEFAULT_OUTPUT_DIR))
        save_folder = osp.join(base_output_dir, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        save_path = osp.join(save_folder, output_name)

    base_name = osp.splitext(osp.basename(save_path))[0]
    track_txt_path = osp.abspath(
        getattr(args, "output_track_txt_path", "") or osp.join(save_folder, f"{base_name}_tracks.txt")
    )
    motion_csv_path = osp.abspath(
        getattr(args, "output_speed_csv_path", "") or osp.join(save_folder, f"{base_name}_motion.csv")
    )
    ensure_parent_dir(track_txt_path)
    ensure_parent_dir(motion_csv_path)

    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        getattr(args, "fps", 30),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {save_path}")

    logger.info(f"save_folder: {save_folder}")
    logger.info(f"video save_path: {save_path}")
    logger.info(f"track result path: {track_txt_path}")
    logger.info(f"motion csv path: {motion_csv_path}")

    return {
        "timestamp": timestamp,
        "save_folder": save_folder,
        "video_path": save_path,
        "track_txt_path": track_txt_path,
        "motion_csv_path": motion_csv_path,
        "writer": writer,
    }


def collect_tracks(online_targets, args, frame_id, results):
    online_tlwhs = []
    online_ids = []
    online_scores = []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / max(tlwh[3], 1e-6) > args.aspect_ratio_thresh

        if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            online_tlwhs.append(np.asarray(tlwh, dtype=np.float32))
            online_ids.append(int(tid))
            online_scores.append(float(target.score))
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
            )

    return online_tlwhs, online_ids, online_scores


def tlwh_to_xyxy(tlwh):
    x1, y1, w, h = tlwh
    return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)


def box_iou(box_a, box_b):
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def match_track_masks(tlwhs, img_info, min_iou=0.1):
    mask_boxes = np.asarray(img_info.get("mask_boxes", []), dtype=np.float32)
    mask_polygons = img_info.get("mask_polygons", [])
    mask_arrays = img_info.get("mask_arrays", [])
    aligned_masks = [{"polygon": None, "box": None, "det_index": None, "iou": 0.0} for _ in range(len(tlwhs))]

    if len(tlwhs) == 0 or len(mask_boxes) == 0 or (not mask_polygons and not mask_arrays):
        return aligned_masks

    matches = []
    for track_idx, tlwh in enumerate(tlwhs):
        track_box = tlwh_to_xyxy(tlwh)
        for det_idx, det_box in enumerate(mask_boxes):
            iou = box_iou(track_box, det_box)
            if iou > 0:
                matches.append((iou, track_idx, det_idx))

    matches.sort(reverse=True)
    assigned_tracks = set()
    assigned_detections = set()

    for iou, track_idx, det_idx in matches:
        if iou < min_iou or track_idx in assigned_tracks or det_idx in assigned_detections:
            continue
        if det_idx >= len(mask_polygons) and det_idx >= len(mask_arrays):
            continue

        aligned_masks[track_idx] = {
            "polygon": mask_polygons[det_idx] if det_idx < len(mask_polygons) else None,
            "mask": mask_arrays[det_idx] if det_idx < len(mask_arrays) else None,
            "box": mask_boxes[det_idx],
            "det_index": det_idx,
            "iou": iou,
        }
        assigned_tracks.add(track_idx)
        assigned_detections.add(det_idx)

    return aligned_masks


def build_tracker(args):
    tracker_type = getattr(args, "tracker_type", "botsort").lower()
    if tracker_type == "bytetrack":
        return BYTETracker(args, frame_rate=args.fps)
    if tracker_type == "botsort":
        return BOTSORT(args, frame_rate=args.fps)
    raise ValueError(f"Unsupported tracker_type: {args.tracker_type}")


def compute_polygon_center(polygon):
    if polygon is None:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or len(points) == 0:
        return None

    if len(points) >= 3:
        moments = cv2.moments(points)
        if abs(moments["m00"]) > 1e-6:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])

    mean_point = points.mean(axis=0)
    return float(mean_point[0]), float(mean_point[1])


def compute_track_center(tlwh, polygon=None):
    polygon_center = compute_polygon_center(polygon)
    if polygon_center is not None:
        return polygon_center

    x1, y1, w, h = tlwh
    return float(x1 + w / 2.0), float(y1 + h / 2.0)


def compute_track_area(tlwh, polygon=None):
    if polygon is not None:
        points = np.asarray(polygon, dtype=np.float32)
        if len(points) >= 3:
            return float(abs(cv2.contourArea(points)))

    return float(max(tlwh[2], 0.0) * max(tlwh[3], 0.0))


def rasterize_polygon_mask(polygon, frame_width, frame_height):
    if polygon is None or frame_width <= 0 or frame_height <= 0:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or len(points) < 3:
        return None

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], 1, lineType=cv2.LINE_8)
    return mask


def compute_mask_edge_margin(mask):
    if mask is None or mask.size == 0:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    height, width = mask.shape[:2]
    return float(min(np.min(xs), np.min(ys), width - 1 - np.max(xs), height - 1 - np.max(ys)))


def is_complete_mask(mask, edge_margin):
    if mask is None or mask.size == 0:
        return False

    margin = max(int(edge_margin), 0)
    if margin == 0:
        return not bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        )

    height, width = mask.shape[:2]
    margin_y = min(margin, height)
    margin_x = min(margin, width)

    return not bool(
        np.any(mask[:margin_y, :])
        or np.any(mask[height - margin_y :, :])
        or np.any(mask[:, :margin_x])
        or np.any(mask[:, width - margin_x :])
    )


def compute_equivalent_diameter(area_real):
    area_real = max(float(area_real), 0.0)
    if area_real <= 0.0:
        return 0.0
    return float(math.sqrt(4.0 * area_real / math.pi))


def compute_turn_angle_deg(point_a, point_b, point_c, min_segment_distance_px=0.0):
    vector_ab = np.asarray(point_b, dtype=np.float32) - np.asarray(point_a, dtype=np.float32)
    vector_bc = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)

    length_ab = float(np.linalg.norm(vector_ab))
    length_bc = float(np.linalg.norm(vector_bc))
    if length_ab <= min_segment_distance_px or length_bc <= min_segment_distance_px:
        return None

    cosine = float(np.dot(vector_ab, vector_bc) / max(length_ab * length_bc, 1e-6))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def build_frame_tracks(tlwhs, track_ids, scores, img_info, args):
    raw_img = img_info.get("raw_img")
    frame_height, frame_width = (raw_img.shape[:2] if raw_img is not None else (0, 0))
    matched_masks = match_track_masks(
        tlwhs,
        img_info,
        min_iou=getattr(args, "mask_match_iou", 0.1),
    )

    frame_tracks = []
    for index, tlwh in enumerate(tlwhs):
        matched = matched_masks[index] if index < len(matched_masks) else {}
        polygon = matched.get("polygon")
        mask = rasterize_polygon_mask(polygon, frame_width, frame_height)
        if mask is None:
            mask = matched.get("mask")
        center = compute_track_center(tlwh, polygon=polygon)
        x1, y1, w, h = [float(value) for value in tlwh]
        x2 = x1 + max(w, 0.0)
        y2 = y1 + max(h, 0.0)
        bbox_edge_margin_px = min(
            x1,
            y1,
            max(frame_width - x2, 0.0),
            max(frame_height - y2, 0.0),
        ) if frame_width > 0 and frame_height > 0 else 0.0
        mask_edge_margin_px = compute_mask_edge_margin(mask)
        edge_margin_px = mask_edge_margin_px if mask_edge_margin_px is not None else bbox_edge_margin_px
        passes_edge_filter = (
            is_complete_mask(mask, getattr(args, "edge_exclusion_margin_px", 0.0))
            if mask is not None
            else edge_margin_px >= max(float(getattr(args, "edge_exclusion_margin_px", 0.0)), 0.0)
        )
        frame_tracks.append(
            {
                "track_id": int(track_ids[index]) if index < len(track_ids) else index + 1,
                "score": float(scores[index]) if index < len(scores) else 0.0,
                "tlwh": np.asarray(tlwh, dtype=np.float32),
                "polygon": polygon,
                "mask": mask,
                "mask_box": matched.get("box"),
                "center": center,
                "trajectory": [],
                "displacement_px": 0.0,
                "displacement_real": 0.0,
                "speed_real_per_s": 0.0,
                "area_px": compute_track_area(tlwh, polygon=polygon),
                "edge_margin_px": float(edge_margin_px),
                "passes_edge_filter": passes_edge_filter,
            }
        )

    return frame_tracks


class MotionAnalyzer:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "enable_motion", True))
        self.frame_interval_seconds = max(float(getattr(args, "frame_interval_seconds", 1.0 / 30.0)), 1e-6)
        self.pixel_to_real_scale = max(float(getattr(args, "pixel_to_real_scale", 1.0)), 0.0)
        self.distance_unit = getattr(args, "distance_unit", "mm") or "unit"
        self.speed_frame_gap = max(int(getattr(args, "speed_frame_gap", 1)), 1)
        self.trail_length = max(int(getattr(args, "trail_length", 24)), 2)
        self.min_track_frames = max(int(getattr(args, "min_track_frames", 5)), 1)
        self.max_turn_angle_deg = float(getattr(args, "max_turn_angle_deg", 120.0))
        self.turn_min_segment_px = max(float(getattr(args, "turn_min_segment_px", 3.0)), 0.0)
        self.edge_exclusion_margin_px = max(float(getattr(args, "edge_exclusion_margin_px", 0.0)), 0.0)
        self.max_history_gap = max(int(getattr(args, "max_history_gap", 10)), 1)
        self.history_size = self.trail_length + self.speed_frame_gap
        self.max_stale_frames = max(self.history_size * 3, 12)
        self.track_states = defaultdict(
            lambda: {
                "history": deque(maxlen=self.history_size),
                "seen_frames": 0,
                "invalid": False,
                "invalid_reason": "",
                "last_turn_angle_deg": None,
                "anchor_center": None,
                "sample_count": 0,
                "speed_sum": 0.0,
                "vx_sum": 0.0,
                "vy_sum": 0.0,
                "rise_speed_sum": 0.0,
            }
        )
        self.csv_rows: List[Dict[str, Any]] = []
        self.accepted_track_ids = set()
        self.area_column_name = self._build_area_column_name()

    def update(self, frame_id, frame_tracks):
        if not self.enabled:
            return self._build_metrics(frame_id, frame_tracks)

        for track in frame_tracks:
            track_id = int(track["track_id"])
            center = tuple(float(value) for value in track["center"])
            state = self.track_states[track_id]
            history = state["history"]

            if len(history) > 0 and (frame_id - history[-1]["frame_id"]) > self.max_history_gap:
                history.clear()
                state["seen_frames"] = 0
                state["anchor_center"] = None
                state["sample_count"] = 0
                state["speed_sum"] = 0.0
                state["vx_sum"] = 0.0
                state["vy_sum"] = 0.0
                state["rise_speed_sum"] = 0.0
                state["last_turn_angle_deg"] = None
                if state["invalid"]:
                    state["invalid"] = False
                    state["invalid_reason"] = ""

            history.append({"frame_id": frame_id, "center": center})
            state["seen_frames"] += 1

            turn_angle_deg = self._compute_recent_turn_angle(history)
            if turn_angle_deg is not None:
                state["last_turn_angle_deg"] = turn_angle_deg
                if turn_angle_deg > self.max_turn_angle_deg and not state["invalid"]:
                    state["invalid"] = True
                    state["invalid_reason"] = "turn_angle"
                    self.accepted_track_ids.discard(track_id)
                    if self.csv_rows:
                        self.csv_rows = [row for row in self.csv_rows if int(row["id"]) != track_id]

            displacement_px = 0.0
            displacement_real = 0.0
            speed_real = 0.0
            delta_x_real = 0.0
            delta_y_real = 0.0
            velocity_x_real = 0.0
            velocity_y_real = 0.0
            rise_displacement = 0.0
            rise_speed = 0.0
            cumulative_dx_real = 0.0
            cumulative_dy_real = 0.0
            is_mature = state["seen_frames"] >= self.min_track_frames
            is_valid = not state["invalid"]
            is_visible = is_mature and is_valid
            edge_margin_px = float(track.get("edge_margin_px", 0.0))
            passes_edge_filter = bool(
                track.get("passes_edge_filter", edge_margin_px >= self.edge_exclusion_margin_px)
            )
            trajectory = [item["center"] for item in list(history)[-self.trail_length:]] if is_visible else []

            if is_visible and state["anchor_center"] is None:
                state["anchor_center"] = center

            if is_visible and len(history) > self.speed_frame_gap:
                reference = history[-1 - self.speed_frame_gap]
                frame_gap = max(frame_id - reference["frame_id"], 1)
                delta_x_px = center[0] - reference["center"][0]
                delta_y_px = center[1] - reference["center"][1]
                delta_x_real = delta_x_px * self.pixel_to_real_scale
                delta_y_real = delta_y_px * self.pixel_to_real_scale
                displacement_px = math.hypot(delta_x_px, delta_y_px)
                displacement_real = displacement_px * self.pixel_to_real_scale
                elapsed_seconds = max(frame_gap * self.frame_interval_seconds, 1e-6)
                speed_real = displacement_real / elapsed_seconds
                velocity_x_real = delta_x_real / elapsed_seconds
                velocity_y_real = delta_y_real / elapsed_seconds
                rise_displacement = max(-delta_y_real, 0.0)
                rise_speed = max(-velocity_y_real, 0.0)

                anchor_center = state["anchor_center"] if state["anchor_center"] is not None else center
                cumulative_dx_real = (center[0] - anchor_center[0]) * self.pixel_to_real_scale
                cumulative_dy_real = (center[1] - anchor_center[1]) * self.pixel_to_real_scale

            track["age_frames"] = state["seen_frames"]
            track["is_mature"] = is_mature
            track["is_valid"] = is_valid
            track["is_visible"] = is_visible
            track["filter_reason"] = state["invalid_reason"]
            track["turn_angle_deg"] = state["last_turn_angle_deg"]
            track["trajectory"] = trajectory
            track["edge_margin_px"] = edge_margin_px
            track["passes_edge_filter"] = passes_edge_filter
            track["displacement_px"] = displacement_px
            track["displacement_real"] = displacement_real
            track["speed_real_per_s"] = speed_real
            track["delta_x_real"] = delta_x_real
            track["delta_y_real"] = delta_y_real
            track["velocity_x_real_per_s"] = velocity_x_real
            track["velocity_y_real_per_s"] = velocity_y_real
            track["rise_displacement"] = rise_displacement
            track["rise_speed_per_s"] = rise_speed
            track["cumulative_dx_real"] = cumulative_dx_real
            track["cumulative_dy_real"] = cumulative_dy_real

            is_csv_sample = is_visible and passes_edge_filter and len(history) > self.speed_frame_gap
            track["is_csv_sample"] = is_csv_sample

            if is_visible:
                self.accepted_track_ids.add(track_id)
            if is_csv_sample:
                area_real = float(track.get("area_px", 0.0)) * (self.pixel_to_real_scale ** 2)
                equivalent_diameter = compute_equivalent_diameter(area_real)
                state["sample_count"] += 1
                state["speed_sum"] += speed_real
                state["vx_sum"] += velocity_x_real
                state["vy_sum"] += velocity_y_real
                state["rise_speed_sum"] += rise_speed
                sample_count = max(state["sample_count"], 1)
                self.csv_rows.append(
                    {
                        "frame": frame_id,
                        "id": track_id,
                        "x": round(center[0], 3),
                        "y": round(center[1], 3),
                        "speed": round(speed_real, 6),
                        "avg_speed": round(state["speed_sum"] / sample_count, 6),
                        self.area_column_name: round(area_real, 6),
                        "equivalent_diameter": round(equivalent_diameter, 9),
                        "vx": round(velocity_x_real, 6),
                        "vy": round(velocity_y_real, 6),
                        "vx_avg": round(state["vx_sum"] / sample_count, 6),
                        "vy_avg": round(state["vy_sum"] / sample_count, 6),
                        "dx_total": round(cumulative_dx_real, 6),
                        "dy_total": round(cumulative_dy_real, 6),
                        "rise_displacement": round(rise_displacement, 6),
                        "rise_speed": round(rise_speed, 6),
                        "rise_speed_avg": round(state["rise_speed_sum"] / sample_count, 6),
                        "rise_total": round(max(-cumulative_dy_real, 0.0), 6),
                        "vertical_speed_abs": round(abs(velocity_y_real), 6),
                        "edge_margin_px": round(edge_margin_px, 3),
                    }
                )

        self._prune_history(frame_id)
        return self._build_metrics(frame_id, frame_tracks)

    def _compute_recent_turn_angle(self, history):
        if len(history) < 3:
            return None

        point_a = history[-3]["center"]
        point_b = history[-2]["center"]
        point_c = history[-1]["center"]
        return compute_turn_angle_deg(
            point_a,
            point_b,
            point_c,
            min_segment_distance_px=self.turn_min_segment_px,
        )

    def _prune_history(self, current_frame_id):
        stale_ids = []
        for track_id, state in self.track_states.items():
            history = state["history"]
            if not history:
                stale_ids.append(track_id)
                continue
            if current_frame_id - history[-1]["frame_id"] > self.max_stale_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.track_states.pop(track_id, None)

    def _build_metrics(self, frame_id, frame_tracks):
        visible_tracks = [track for track in frame_tracks if track.get("is_visible", True)]
        speeds = [float(track.get("speed_real_per_s", 0.0)) for track in visible_tracks]
        areas = [float(track.get("area_px", 0.0)) for track in visible_tracks]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        avg_area = float(np.mean(areas)) if areas else 0.0

        return {
            "frame_id": frame_id,
            "tracked_count": len(visible_tracks),
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "avg_area_px": avg_area,
            "distance_unit": self.distance_unit,
            "samples_written": len(self.csv_rows),
            "unique_tracks": len(self.accepted_track_ids),
        }

    def _build_area_column_name(self):
        normalized_unit = re.sub(r"[^0-9A-Za-z]+", "_", str(self.distance_unit).strip()).strip("_").lower()
        return f"area_{normalized_unit}2" if normalized_unit else "area_real2"

    def write_csv(self, csv_path):
        ensure_parent_dir(csv_path)
        fieldnames = [
            "frame",
            "id",
            "x",
            "y",
            "speed",
            "avg_speed",
            self.area_column_name,
            "equivalent_diameter",
            "vx",
            "vy",
            "vx_avg",
            "vy_avg",
            "dx_total",
            "dy_total",
            "rise_displacement",
            "rise_speed",
            "rise_speed_avg",
            "rise_total",
            "vertical_speed_abs",
            "edge_margin_px",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_rows)

        logger.info(f"save motion csv to {csv_path}")

    def summary(self):
        return {
            "motion_samples": len(self.csv_rows),
            "unique_tracks": len(self.accepted_track_ids),
            "distance_unit": self.distance_unit,
        }


def render_mask_frame(raw_img, frame_tracks, frame_id, fps, args):
    visible_tracks = [track for track in frame_tracks if track.get("is_csv_sample", False)]
    return plot_tracking(
        raw_img,
        [track["tlwh"] for track in visible_tracks],
        [track["track_id"] for track in visible_tracks],
        [track["score"] for track in visible_tracks],
        frame_id=frame_id + 1,
        fps=fps,
        mask_polygons=[track["polygon"] for track in visible_tracks],
        trajectories=[track["trajectory"] for track in visible_tracks],
        centers=[track["center"] for track in visible_tracks],
        speeds=[track["speed_real_per_s"] for track in visible_tracks],
        show_speed_text=getattr(args, "show_speed_text", False),
        speed_unit=getattr(args, "distance_unit", "unit"),
        mode="mask",
    )


def build_default_metrics(frame_id, fps, distance_unit="unit"):
    return {
        "frame_id": frame_id,
        "fps": fps,
        "tracked_count": 0,
        "avg_speed": 0.0,
        "max_speed": 0.0,
        "avg_area_px": 0.0,
        "distance_unit": distance_unit,
        "samples_written": 0,
        "unique_tracks": 0,
    }


def process_frame(frame, predictor, tracker, timer, args, frame_id, results, motion_analyzer=None):
    outputs, img_info = predictor.predict(frame, timer)

    if outputs is None:
        timer.stop()
        frame_tracks = []
        frame_metrics = (
            motion_analyzer.update(frame_id + 1, frame_tracks)
            if motion_analyzer is not None
            else build_default_metrics(frame_id + 1, 0.0, getattr(args, "distance_unit", "unit"))
        )
        frame_metrics["fps"] = 0.0

        return (
            render_mask_frame(img_info["raw_img"], frame_tracks, frame_id, 0.0, args),
            frame_tracks,
            frame_metrics,
        )

    if getattr(args, "tracker_type", "botsort").lower() == "botsort":
        online_targets = tracker.update(
            outputs,
            img_info["raw_img"],
            img_size=args.input_size,
        )
    else:
        online_targets = tracker.update(
            outputs,
            [img_info["height"], img_info["width"]],
            img_size=args.input_size,
        )

    online_tlwhs, online_ids, online_scores = collect_tracks(online_targets, args, frame_id, results)
    timer.stop()

    frame_tracks = build_frame_tracks(online_tlwhs, online_ids, online_scores, img_info, args)
    frame_metrics = (
        motion_analyzer.update(frame_id + 1, frame_tracks)
        if motion_analyzer is not None
        else build_default_metrics(frame_id + 1, 0.0, getattr(args, "distance_unit", "unit"))
    )
    frame_metrics["fps"] = 1.0 / max(1e-5, timer.average_time)

    return (
        render_mask_frame(
            img_info["raw_img"],
            frame_tracks,
            frame_id,
            frame_metrics["fps"],
            args,
        ),
        frame_tracks,
        frame_metrics,
    )


def save_results(result_path, results):
    ensure_parent_dir(result_path)
    with open(result_path, "w", encoding="utf-8") as file:
        file.writelines(results)
    logger.info(f"save track results to {result_path}")


def infer_demo_mode(args):
    if getattr(args, "demo", "").lower() == "camera":
        return "camera"
    if osp.isdir(args.path):
        return "images"
    if osp.splitext(args.path)[1].lower() in IMAGE_EXT:
        return "images"
    return "video"


def should_emit_progress(frame_index, total_frames, args):
    stride = max(int(getattr(args, "preview_stride", 5)), 1)
    is_last = total_frames is not None and frame_index + 1 >= total_frames
    return frame_index == 0 or (frame_index + 1) % stride == 0 or is_last


def emit_progress(progress_callback, preview_frame, metrics, output_info, frame_index, total_frames, source_path):
    if progress_callback is None:
        return

    progress_callback(
        {
            "frame_id": frame_index + 1,
            "total_frames": total_frames,
            "preview_frame": preview_frame,
            "metrics": metrics,
            "output_video_path": output_info["video_path"],
            "output_track_txt_path": output_info["track_txt_path"],
            "output_speed_csv_path": output_info["motion_csv_path"],
            "source_path": source_path,
        }
    )


def finalize_outputs(output_info, results, motion_analyzer):
    output_info["writer"].release()
    save_results(output_info["track_txt_path"], results)
    if motion_analyzer is not None:
        motion_analyzer.write_csv(output_info["motion_csv_path"])


def imageflow_demo(predictor, current_time, args, progress_callback=None, stop_event=None):
    if infer_demo_mode(args) == "camera":
        cap = cv2.VideoCapture(args.camid)
        source_name = "camera.mp4"
        source_path = str(args.camid)
    else:
        cap = cv2.VideoCapture(args.path)
        source_name = osp.splitext(osp.basename(args.path))[0] + ".mp4"
        source_path = args.path

    if not cap.isOpened():
        raise ValueError(f"Cannot open input source: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if source_fps and source_fps > 0:
        args.fps = source_fps
        args.frame_interval_seconds = 1.0 / source_fps

    output_info = prepare_output(args, current_time, source_name, (width, height))
    tracker = build_tracker(args)
    timer = MyTimer()
    motion_analyzer = MotionAnalyzer(args)
    frame_id = 0
    results = []
    stopped = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            if frame_id % args.counts == 0:
                logger.info("Processing frame {} ({:.2f} fps)".format(frame_id, 1.0 / max(1e-5, timer.average_time)))

            ret_val, frame = cap.read()
            if not ret_val:
                break

            online_im, _, frame_metrics = process_frame(
                frame,
                predictor,
                tracker,
                timer,
                args,
                frame_id,
                results,
                motion_analyzer=motion_analyzer,
            )
            output_info["writer"].write(online_im)

            if should_emit_progress(frame_id, total_frames, args):
                emit_progress(progress_callback, online_im, frame_metrics, output_info, frame_id, total_frames, source_path)

            frame_id += 1
    finally:
        cap.release()

    finalize_outputs(output_info, results, motion_analyzer)
    summary = motion_analyzer.summary()
    summary.update(
        {
            "stopped": stopped,
            "processed_frames": frame_id,
            "total_frames": total_frames,
            "video_path": output_info["video_path"],
            "track_txt_path": output_info["track_txt_path"],
            "motion_csv_path": output_info["motion_csv_path"],
            "save_folder": output_info["save_folder"],
            "source_path": source_path,
        }
    )
    return summary


def image_sequence_demo(predictor, current_time, args, progress_callback=None, stop_event=None):
    image_paths = get_image_list(args.path)
    if not image_paths:
        raise ValueError(f"No images found in: {args.path}")

    first_frame = read_image(image_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read image: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    folder_name = osp.basename(osp.normpath(args.path)) or "images"
    output_name = f"{folder_name}.mp4"
    total_frames = len(image_paths)
    output_info = prepare_output(args, current_time, output_name, (width, height))

    tracker = build_tracker(args)
    timer = MyTimer()
    motion_analyzer = MotionAnalyzer(args)
    results = []
    stopped = False
    processed_frames = 0

    try:
        for frame_id, image_path in enumerate(image_paths):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            if frame_id % args.counts == 0:
                logger.info(
                    "Processing frame {} ({:.2f} fps): {}".format(
                        frame_id,
                        1.0 / max(1e-5, timer.average_time),
                        image_path,
                    )
                )

            frame = read_image(image_path)
            if frame is None:
                logger.warning(f"Skip unreadable image: {image_path}")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            online_im, _, frame_metrics = process_frame(
                frame,
                predictor,
                tracker,
                timer,
                args,
                frame_id,
                results,
                motion_analyzer=motion_analyzer,
            )
            output_info["writer"].write(online_im)

            if should_emit_progress(frame_id, total_frames, args):
                emit_progress(progress_callback, online_im, frame_metrics, output_info, frame_id, total_frames, args.path)

            processed_frames = frame_id + 1
    finally:
        pass

    finalize_outputs(output_info, results, motion_analyzer)
    summary = motion_analyzer.summary()
    summary.update(
        {
            "stopped": stopped,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "video_path": output_info["video_path"],
            "track_txt_path": output_info["track_txt_path"],
            "motion_csv_path": output_info["motion_csv_path"],
            "save_folder": output_info["save_folder"],
            "source_path": args.path,
        }
    )
    return summary


def main(args, progress_callback=None, stop_event=None):
    logger.info(f"args.model_path: {args.model_path}")
    logger.info(f"args.device: {getattr(args, 'device', 'auto')}")
    args.demo = infer_demo_mode(args)
    predictor = create_predictor(args)
    current_time = time.localtime()

    if args.demo == "images":
        return image_sequence_demo(predictor, current_time, args, progress_callback=progress_callback, stop_event=stop_event)
    return imageflow_demo(predictor, current_time, args, progress_callback=progress_callback, stop_event=stop_event)


class Args:
    demo = "video" if osp.isfile(DEFAULT_VIDEO_PATH) else "images"
    path = DEFAULT_VIDEO_PATH if osp.isfile(DEFAULT_VIDEO_PATH) else DEFAULT_IMAGE_DIR
    save_result = DEFAULT_OUTPUT_DIR
    output_video_path = ""
    output_track_txt_path = ""
    output_speed_csv_path = ""
    fps = 30
    counts = 30
    camid = 0

    model_path = osp.join(ROOT_DIR, "3900.pt")
    input_size = (640, 640)
    fp16 = False
    predict_conf = 0.1
    predict_iou = 0.45

    tracker_type = "botsort"
    track_thresh = 0.6
    track_high_thresh = 0.6
    track_low_thresh = 0.1
    new_track_thresh = 0.7
    track_buffer = 30
    match_thresh = 0.8
    min_box_area = 16
    aspect_ratio_thresh = 10
    proximity_thresh = 0.5
    appearance_thresh = 0.25
    cmc_method = "sparseOptFlow"
    with_reid = False
    fast_reid_config = osp.join(ROOT_DIR, "BoT-SORT", "fast_reid", "configs", "MOT17", "sbs_S50.yml")
    fast_reid_weights = osp.join(ROOT_DIR, "BoT-SORT", "pretrained", "mot17_sbs_S50.pth")
    device = "auto"
    mot20 = False
    ablation = False
    name = "custom"
    mask_match_iou = 0.1

    enable_motion = True
    frame_interval_seconds = 1.0 / 30.0
    pixel_to_real_scale = 1.0
    distance_unit = "mm"
    speed_frame_gap = 1
    min_track_frames = 5
    max_turn_angle_deg = 120.0
    turn_min_segment_px = 3.0
    edge_exclusion_margin_px = 0.0
    max_history_gap = 10
    trail_length = 24
    show_speed_text = False
    preview_stride = 5


if __name__ == "__main__":
    args = Args()
    main(args)
