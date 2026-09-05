#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from dataclasses import dataclass, field
import math
import os
import os.path as osp
import re
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .gas_holdup import EdgeBubblePolicy, GasHoldupResult, ROI, analyze_masks
from .image_io import get_image_list, read_image, write_image
from .inference import create_predictor


CSV_COLUMNS = (
    "Frame",
    "BubbleCount",
    "BubbleArea",
    "ROIArea",
    "BubbleAreaRatio",
    "GasHoldup",
    "AverageDiameterPx",
    "AverageDiameter",
    "DiameterUnit",
    "RealUnitsPerPixel",
    "ProcessingTime",
)


@dataclass
class GasHoldupConfig:
    model_path: str
    input_folder: str
    output_overlay_folder: str
    output_csv: str
    confidence_threshold: float = 0.1
    min_bubble_area: int = 10
    roi: ROI = field(default_factory=ROI)
    edge_policy: EdgeBubblePolicy = EdgeBubblePolicy.INCLUDE
    calibration_pixels: float = 100.0
    calibration_length: float = 1.0
    distance_unit: str = "mm"
    input_size: Tuple[int, int] = (640, 640)
    high_quality_segmentation: bool = False
    predict_iou: float = 0.45
    device: str = "auto"

    @property
    def predict_conf(self):
        return self.confidence_threshold

    @property
    def real_units_per_pixel(self):
        calibration_pixels = float(self.calibration_pixels)
        calibration_length = float(self.calibration_length)
        if not math.isfinite(calibration_pixels) or calibration_pixels <= 0:
            raise ValueError("Calibration Pixels must be greater than 0.")
        if not math.isfinite(calibration_length) or calibration_length <= 0:
            raise ValueError("Calibration Length must be greater than 0.")
        return calibration_length / calibration_pixels

    @property
    def normalized_distance_unit(self):
        unit = str(self.distance_unit).strip()
        if re.fullmatch(r"[A-Za-zµμ]{1,12}", unit) is None:
            raise ValueError("Distance Unit must contain 1-12 letters, such as mm or µm.")
        return unit


def _extract_masks(image_info, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    masks = []
    for source_mask in image_info.get("mask_arrays", []):
        mask = np.asarray(source_mask)
        if mask.shape[:2] != (frame_height, frame_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            )
        masks.append(mask.astype(np.uint8, copy=False))

    if masks:
        return masks

    for polygon in image_info.get("mask_polygons", []):
        if polygon is None or len(polygon) < 3:
            continue
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        points = np.rint(np.asarray(polygon)).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
    return masks


def render_gas_holdup_overlay(
    frame,
    result: GasHoldupResult,
    average_diameter,
    distance_unit,
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rendered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    palette = (
        (36, 181, 255),
        (255, 151, 64),
        (96, 220, 128),
        (214, 120, 255),
        (255, 214, 92),
    )

    for index, mask in enumerate(result.accepted_masks):
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        selected = mask.astype(bool)
        rendered[selected] = (
            rendered[selected].astype(np.float32) * 0.45 + color * 0.55
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(rendered, contours, -1, tuple(int(value) for value in color), 1)

    x1, y1, x2, y2 = result.roi_bounds
    cv2.rectangle(rendered, (x1, y1), (max(x2 - 1, x1), max(y2 - 1, y1)), (255, 255, 255), 1)
    lines = (
        f"Gas Holdup : {result.gas_holdup:.2f} %",
        f"Bubble Count : {result.bubble_count}",
        f"Bubble Area : {result.bubble_area} px",
        f"Average Diameter : {average_diameter:.4f} {distance_unit}",
        f"Pixel Diameter : {result.average_diameter:.2f} px",
    )
    text_height = 22 * len(lines) + 12
    panel_width = min(max(frame.shape[1] - 12, 1), 390)
    cv2.rectangle(rendered, (6, 6), (panel_width, min(text_height, frame.shape[0] - 1)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            rendered,
            line,
            (14, 28 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _csv_row(
    frame_id,
    result,
    average_diameter,
    distance_unit,
    real_units_per_pixel,
    processing_time,
):
    return {
        "Frame": frame_id,
        "BubbleCount": result.bubble_count,
        "BubbleArea": result.bubble_area,
        "ROIArea": result.roi_area,
        "BubbleAreaRatio": f"{result.bubble_area_ratio:.6f}",
        "GasHoldup": f"{result.gas_holdup:.6f}",
        "AverageDiameterPx": f"{result.average_diameter:.6f}",
        "AverageDiameter": f"{average_diameter:.6f}",
        "DiameterUnit": distance_unit,
        "RealUnitsPerPixel": f"{real_units_per_pixel:.9f}",
        "ProcessingTime": f"{processing_time:.6f}",
    }


def run_gas_holdup(
    config: GasHoldupConfig,
    progress_callback: Optional[Callable[[dict], None]] = None,
    stop_event=None,
    predictor_factory=create_predictor,
):
    input_root = osp.normcase(osp.realpath(config.input_folder))
    overlay_root = osp.normcase(osp.realpath(config.output_overlay_folder))
    try:
        overlay_inside_input = osp.commonpath((input_root, overlay_root)) == input_root
    except ValueError:
        overlay_inside_input = False
    if overlay_inside_input:
        raise ValueError("Output Overlay Folder must be outside the Input Folder.")
    if not 0.0 <= float(config.confidence_threshold) <= 1.0:
        raise ValueError("Confidence Threshold must be between 0 and 1.")
    if int(config.min_bubble_area) < 0:
        raise ValueError("Minimum Bubble Area must not be negative.")
    real_units_per_pixel = config.real_units_per_pixel
    distance_unit = config.normalized_distance_unit

    image_paths = get_image_list(config.input_folder)
    if not image_paths:
        raise ValueError(f"No images found in: {config.input_folder}")

    os.makedirs(config.output_overlay_folder, exist_ok=True)
    csv_parent = osp.dirname(osp.abspath(config.output_csv))
    os.makedirs(csv_parent, exist_ok=True)
    predictor = predictor_factory(config)
    if str(getattr(predictor, "task", "segment")).lower() != "segment":
        raise ValueError("Gas Holdup requires a YOLO instance-segmentation model.")

    processed_frames = 0
    stopped = False
    with open(config.output_csv, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for frame_index, image_path in enumerate(image_paths):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            started_at = time.perf_counter()
            frame = read_image(image_path)
            if frame is None:
                raise ValueError(f"Cannot read image: {image_path}")
            _, image_info = predictor.predict(frame)
            masks = _extract_masks(image_info, frame.shape)
            result = analyze_masks(
                masks,
                frame_shape=frame.shape,
                min_bubble_area=config.min_bubble_area,
                roi=config.roi,
                edge_policy=config.edge_policy,
            )
            average_diameter = result.average_diameter * real_units_per_pixel
            overlay = render_gas_holdup_overlay(
                frame,
                result,
                average_diameter,
                distance_unit,
            )
            overlay_name = f"{frame_index + 1:06d}_{osp.splitext(osp.basename(image_path))[0]}.png"
            overlay_path = osp.join(config.output_overlay_folder, overlay_name)
            if not write_image(overlay_path, overlay):
                raise OSError(f"Cannot save overlay image: {overlay_path}")

            processing_time = time.perf_counter() - started_at
            writer.writerow(
                _csv_row(
                    frame_index + 1,
                    result,
                    average_diameter,
                    distance_unit,
                    real_units_per_pixel,
                    processing_time,
                )
            )
            csv_file.flush()
            processed_frames += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "frame_id": frame_index + 1,
                        "total_frames": len(image_paths),
                        "current_frame": osp.basename(image_path),
                        "preview_frame": overlay,
                        "metrics": {
                            "bubble_count": result.bubble_count,
                            "bubble_area": result.bubble_area,
                            "roi_area": result.roi_area,
                            "bubble_area_ratio": result.bubble_area_ratio,
                            "gas_holdup": result.gas_holdup,
                            "average_diameter_px": result.average_diameter,
                            "average_diameter": average_diameter,
                            "distance_unit": distance_unit,
                            "real_units_per_pixel": real_units_per_pixel,
                            "processing_time": processing_time,
                            "fps": 1.0 / max(processing_time, 1e-9),
                        },
                        "overlay_path": overlay_path,
                        "csv_path": config.output_csv,
                    }
                )

    return {
        "stopped": stopped,
        "processed_frames": processed_frames,
        "total_frames": len(image_paths),
        "overlay_folder": osp.abspath(config.output_overlay_folder),
        "csv_path": osp.abspath(config.output_csv),
        "source_path": osp.abspath(config.input_folder),
    }
