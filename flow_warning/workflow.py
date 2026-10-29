import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from bubble_analyzer.gas_holdup import ROI
from bubble_analyzer.image_io import get_image_list, read_image, write_image
from bubble_analyzer.inference import create_predictor
from .measurements import FrameMeasurer
from .rules import GROUPS, WarningEngine, WarningSettings, aggregate_window, fit_baseline


@dataclass
class FlowWarningConfig:
    model_path: str
    baseline_path: str
    input_path: str
    output_dir: str
    settings: WarningSettings = field(default_factory=WarningSettings)
    roi: ROI = field(default_factory=ROI)
    min_area: int = 10
    max_distance: float = 40
    predict_conf: float = 0.1
    predict_iou: float = 0.45
    input_size: tuple = (640, 640)
    high_quality_segmentation: bool = True
    device: str = "auto"
    distance_unit: str = "mm"

    def validate(self):
        self.settings.validate()
        if not Path(self.model_path).is_file():
            raise ValueError("请选择存在的气泡实例分割模型。")
        if not self.output_dir.strip():
            raise ValueError("请选择输出目录。")
        sources = [Path(value).resolve() for value in (self.baseline_path, self.input_path)]
        if any(not source.exists() for source in sources):
            raise ValueError("正常基线或待分析来源不存在。")
        if sources[0] == sources[1] or sources[0] in sources[1].parents or sources[1] in sources[0].parents:
            raise ValueError("正常基线与待分析数据应独立，不能使用相同或互相包含的来源。")
        output = Path(self.output_dir).resolve()
        if any(source.is_dir() and (source == output or source in output.parents) for source in sources):
            raise ValueError("输出目录必须位于两个输入目录之外。")
        if not math.isfinite(self.max_distance) or self.max_distance <= 0 or self.min_area < 1:
            raise ValueError("关联距离和最小面积必须为正数。")
        if not 0 <= self.predict_conf <= 1 or not 0 <= self.predict_iou <= 1:
            raise ValueError("分割置信度和 IoU 必须在 0 到 1 之间。")
        if not re.fullmatch(r"[A-Za-zµμ]{1,12}", self.distance_unit):
            raise ValueError("长度单位应为 mm、cm 等字母标签。")


class FrameSource:
    def __init__(self, path):
        self.path = Path(path)
        self.images = get_image_list(str(path)) if self.path.is_dir() else None
        self.total = len(self.images) if self.images is not None else 0
        if self.images is not None and not self.images:
            raise ValueError(f"没有找到图像：{path}")

    def __iter__(self):
        if self.images is not None:
            for frame_id, path in enumerate(self.images, 1):
                try:
                    frame = read_image(path)
                except (OSError, ValueError, cv2.error):
                    frame = None
                yield frame_id, str(path), frame
            return
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"无法打开视频：{self.path}")
        self.total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        try:
            frame_id = 1
            while True:
                ok, frame = capture.read()
                if not ok:
                    if self.total and frame_id <= self.total:
                        raise ValueError(f"视频在第 {frame_id} 帧提前解码失败，请检查输入文件。")
                    break
                yield frame_id, f"{self.path}#{frame_id}", frame
                frame_id += 1
        finally:
            capture.release()


def extract_masks(info, shape):
    masks = info.get("mask_arrays", [])
    if masks:
        return [np.asarray(mask, dtype=np.uint8) for mask in masks]
    masks = []
    for polygon in info.get("mask_polygons", []):
        if polygon is None or len(polygon) < 3:
            continue
        mask = np.zeros(shape[:2], dtype=np.uint8)
        points = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
    return masks


def render_overlay(frame, masks, measurement, roi, status):
    output = frame.copy()
    x1, y1, x2, y2 = roi.resolve(frame.shape)
    if measurement.valid:
        tint = output.copy()
        for mask in masks:
            if mask.shape == frame.shape[:2]:
                region = tint[y1:y2, x1:x2]
                region[mask[y1:y2, x1:x2].astype(bool)] = (80, 190, 245)
        output = cv2.addWeighted(output, 0.65, tint, 0.35, 0)
        for sample in measurement.samples:
            center = (round(sample["x"]), round(sample["y"]))
            color = (80, 220, 120) if not sample["reason"] else (60, 170, 255)
            cv2.circle(output, center, 3, color, -1)
            if sample["track_id"] is not None:
                cv2.putText(output, str(sample["track_id"]), center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (230, 180, 60), 1)
    text = f"Frame {measurement.frame_id} | {status}"
    if measurement.reason:
        text += f" | {measurement.reason}"
    cv2.putText(output, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    return output


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_flow_warning(config, progress_callback=None, stop_event=None, predictor_factory=create_predictor):
    config.validate()
    sources = {"baseline": FrameSource(config.baseline_path), "analysis": FrameSource(config.input_path)}
    predictor = predictor_factory(config)
    if getattr(predictor, "task", None) != "segment":
        raise ValueError("流态预警需要气泡实例分割模型。")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("flow_warning_%Y%m%d_%H%M%S_"), dir=config.output_dir))
    overlays = output / "overlays"
    overlays.mkdir()
    settings = config.settings
    expected_shape = None
    baseline_frames = []
    baseline = None
    engine = None
    last_window = None
    processed = {"baseline": 0, "analysis": 0}
    stopped = False
    frame_columns = [
        "phase", "frame", "time", "source", "valid", "reason", "bubble_count",
        "area_ratio", "complete_count", "size_samples", "motion_samples",
    ]
    track_columns = [
        "phase", "source", "frame", "time", "track_id", "x", "y", "area_px",
        "diameter", "rise_velocity", "vx", "turn_angle", "reason",
    ]
    metric_columns = ["area_ratio", "bubble_count", "large_fraction", "d90", "rise_median", "velocity_iqr", "down_fraction"]
    window_columns = [
        "start_frame", "end_frame", "start_time", "end_time", "valid_fraction", "complete",
        "size_samples", "motion_samples", "motion_coverage", "state",
    ] + metric_columns + [f"{group}_{suffix}" for group in GROUPS for suffix in ("state", "reason")]

    def emit(payload):
        if progress_callback is not None:
            progress_callback(payload)

    def evaluate(pending, writer):
        nonlocal last_window
        window = engine.update(aggregate_window(pending, settings, baseline["diameter_cutoff"]))
        last_window = window
        row = {key: window[key] for key in window_columns if key in window}
        row.update(window["metrics"])
        for group, state in window["groups"].items():
            row[f"{group}_state"] = state["state"]
            row[f"{group}_reason"] = state["reason"]
        writer.writerow(row)
        return window

    with (output / "frames.csv").open("w", newline="", encoding="utf-8-sig") as frame_file, \
            (output / "tracks.csv").open("w", newline="", encoding="utf-8-sig") as track_file, \
            (output / "windows.csv").open("w", newline="", encoding="utf-8-sig") as window_file:
        frame_writer = csv.DictWriter(frame_file, fieldnames=frame_columns)
        track_writer = csv.DictWriter(track_file, fieldnames=track_columns, extrasaction="ignore")
        window_writer = csv.DictWriter(window_file, fieldnames=window_columns)
        for writer in (frame_writer, track_writer, window_writer):
            writer.writeheader()
        for phase, source in sources.items():
            measurer = FrameMeasurer(config.roi, settings.pixel_scale, config.min_area, config.max_distance)
            pending = []
            iterator = iter(source)
            try:
                for frame_id, path, frame in iterator:
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        break
                    if frame is not None and expected_shape is None:
                        expected_shape = frame.shape[:2]
                        config.roi.resolve(expected_shape)
                    error = "unreadable_image" if frame is None else ""
                    if frame is not None and frame.shape[:2] != expected_shape:
                        error = "frame_shape_changed"
                    masks = []
                    if not error:
                        detections, info = predictor.predict(frame)
                        masks = extract_masks(info, frame.shape)
                        boxes = info.get("mask_boxes", [])
                        detection_count = len(boxes) if len(boxes) else (len(detections) if detections is not None else 0)
                        if detection_count > len(masks):
                            error = "missing_segmentation_masks"
                    shape = expected_shape or (2, 2)
                    measured = measurer.measure(masks, shape, frame_id, (frame_id - 1) / settings.fps, error=error)
                    measured.source = path
                    frame_writer.writerow({
                        "phase": phase, "frame": frame_id, "time": measured.timestamp, "source": path,
                        "valid": measured.valid, "reason": measured.reason, "bubble_count": measured.bubble_count,
                        "area_ratio": measured.area_ratio, "complete_count": measured.complete_count,
                        "size_samples": len(measured.diameters), "motion_samples": len(measured.velocities),
                    })
                    for sample in measured.samples:
                        track_writer.writerow({**sample, "phase": phase, "source": path})
                    processed[phase] += 1
                    payload = {"phase": phase, "frame_id": frame_id, "total_frames": source.total, "current_frame": path}
                    if phase == "baseline":
                        measured.samples = []
                        baseline_frames.append(measured)
                    else:
                        pending.append(measured)
                        window = evaluate(pending, window_writer) if len(pending) == settings.frames_per_window else None
                        if window is not None:
                            pending = []
                            payload["window"] = window
                            payload["events"] = [dict(event) for event in engine.events]
                            window_file.flush()
                        status = last_window["state"] if last_window else "collecting"
                        canvas = frame if frame is not None and frame.shape[:2] == shape else np.zeros((*shape, 3), np.uint8)
                        overlay = render_overlay(canvas, masks, measured, config.roi, status)
                        overlay_path = overlays / f"{frame_id:06d}.png"
                        if not write_image(str(overlay_path), overlay):
                            raise OSError(f"无法保存证据图像：{overlay_path}")
                        measured.samples = []
                        payload.update(overlay_path=str(overlay_path), output_dir=str(output))
                    emit(payload)
                    frame_file.flush()
                    track_file.flush()
            finally:
                iterator.close()
            if phase == "analysis" and pending:
                window = evaluate(pending, window_writer)
                emit({"phase": "analysis", "frame_id": processed[phase], "total_frames": source.total,
                      "window": window, "events": [dict(event) for event in engine.events]})
            if stopped:
                break
            if phase == "baseline":
                baseline = fit_baseline(baseline_frames, settings)
                baseline.update(config=asdict(config), frame_shape=list(expected_shape))
                _write_json(output / "baseline.json", baseline)
                engine = WarningEngine(baseline, settings)
                emit({"phase": "baseline_ready", "frame_id": processed[phase], "baseline": baseline})
                baseline_frames.clear()
            elif not processed[phase]:
                raise ValueError("待测来源没有可解码的视频帧。")
    events = engine.events if engine is not None else []
    if engine is not None and last_window is not None:
        engine.finish(last_window, "stopped" if stopped else "end_of_sequence")
    for event in events:
        event["evidence_path"] = str(overlays / f"{event['evidence_frame']:06d}.png")
    _write_json(output / "events.json", events)
    event_columns = ["id", "group", "start_frame", "start_time", "trigger_frame", "trigger_time", "end_frame", "end_time",
                     "end_reason", "reason", "evidence_frame", "evidence_path"]
    with (output / "events.csv").open("w", newline="", encoding="utf-8-sig") as event_file:
        writer = csv.DictWriter(event_file, fieldnames=event_columns)
        writer.writeheader()
        writer.writerows(events)
    return {"output_dir": str(output), "processed_frames": processed["analysis"], "baseline_frames": processed["baseline"],
            "stopped": stopped, "events": events, "baseline": baseline}
