#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Segment bubbles in one image with a YOLO instance-segmentation model.

Example:
    python segment_bubbles.py data/dataset/test_images/frame.png --model 3900.pt

The output directory contains an annotated image, a combined binary mask,
one binary mask per bubble, transparent bubble cutouts, and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(SCRIPT_DIR / ".ultralytics"))

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def ensure_custom_addmodules() -> None:
    """Register the custom layer used by the project's 3900*.pt weights."""
    module_name = "ultralytics.nn.Addmodules"
    bifpn_name = f"{module_name}.BiFPN"
    if bifpn_name in sys.modules:
        return
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return

    addmodules_package = types.ModuleType(module_name)
    bifpn_module = types.ModuleType(bifpn_name)

    class Swish(nn.Module):
        def forward(self, value):
            return value * torch.sigmoid(value)

    class BiFPN(nn.Module):
        def __init__(self, weight_num=2, *unused_args, **unused_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(int(weight_num), dtype=torch.float32))
            self.epsilon = 1e-4
            self.swish = Swish()

        def forward(self, values):
            if not isinstance(values, (list, tuple)) or not values:
                return values
            weights = self.weight
            if weights.numel() > len(values):
                weights = weights[: len(values)]
            elif weights.numel() < len(values):
                padding = torch.ones(
                    len(values) - weights.numel(), device=weights.device, dtype=weights.dtype
                )
                weights = torch.cat([weights, padding], dim=0)
            weights = self.swish(weights)
            weights = weights / (torch.sum(weights, dim=0) + self.epsilon)
            fused = values[0] * weights[0]
            for index in range(1, len(values)):
                fused = fused + values[index] * weights[index]
            return fused

    bifpn_module.swish = Swish
    bifpn_module.Bi_FPN = BiFPN
    addmodules_package.BiFPN = bifpn_module
    sys.modules[module_name] = addmodules_package
    sys.modules[bifpn_name] = bifpn_module
    try:
        import ultralytics.nn as ultralytics_nn

        ultralytics_nn.Addmodules = addmodules_package
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 YOLO 实例分割模型分割单张图片中的气泡。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="输入图片路径")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO 实例分割权重；未指定时尝试脚本目录下的 3900.pt、3900HD.pt、best.pt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出目录；默认是输入文件名后加 _segmented",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备，例如 auto、cpu、cuda:0；auto 交给 Ultralytics 自动选择",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="掩膜二值化阈值")
    parser.add_argument(
        "--classes",
        default="",
        help="只保留指定类别 ID，多个 ID 用逗号分隔，例如 0,2；留空表示保留全部",
    )
    parser.add_argument("--no-crops", action="store_true", help="不保存逐气泡透明裁剪图")
    parser.add_argument("--no-overlay", action="store_true", help="不保存标注覆盖图")
    parser.add_argument("--no-json", action="store_true", help="不保存 JSON 结果")
    return parser


def choose_model(model_arg: Path | None) -> Path:
    if model_arg is not None:
        model_path = model_arg.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        return model_path

    for filename in ("3900.pt", "3900HD.pt", "best.pt"):
        candidate = SCRIPT_DIR / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到默认模型，请使用 --model 指定 YOLO 实例分割权重。")


def parse_class_ids(value: str) -> set[int] | None:
    value = value.strip()
    if not value:
        return None
    class_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            class_id = int(item)
        except ValueError as exc:
            raise ValueError(f"类别 ID 必须是整数: {item}") from exc
        if class_id < 0:
            raise ValueError(f"类别 ID 不能为负数: {class_id}")
        class_ids.add(class_id)
    return class_ids or None


def read_image(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"输入图片不存在: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片，请检查格式: {path}")
    return image


def write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    if suffix not in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        suffix = ".png"
    encoded = cv2.imencode(suffix, image)[1]
    encoded.tofile(str(path))


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def resize_mask(mask, height: int, width: int, threshold: float):
    mask = as_numpy(mask).astype(np.float32, copy=False)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return mask >= threshold


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def clipped_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def color_for(index: int):
    return (
        int((37 * index + 80) % 256),
        int((67 * index + 150) % 256),
        int((97 * index + 220) % 256),
    )


def draw_overlay(image, detections):
    overlay = image.copy()
    height, width = image.shape[:2]
    for index, detection in enumerate(detections, start=1):
        mask = detection["mask"]
        color = np.asarray(color_for(index), dtype=np.float32)
        pixels = overlay[mask].astype(np.float32)
        overlay[mask] = (pixels * 0.45 + color * 0.55).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_for(index), 2)

        x1, y1, x2, y2 = detection["box_xyxy"]
        cv2.rectangle(overlay, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color_for(index), 1)
        label = f'{index}: {detection["class_name"]} {detection["confidence"]:.2f}'
        text_y = max(18, min(height - 4, y1 - 5 if y1 > 22 else y2 + 18))
        cv2.putText(
            overlay,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_for(index),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_detection_outputs(image, detection, output_dir: Path, index: int, save_crop: bool) -> None:
    mask = detection["mask"]
    mask_path = output_dir / "masks" / f"mask_{index:03d}.png"
    write_image(mask_path, mask.astype(np.uint8) * 255)
    detection["mask_path"] = str(mask_path.relative_to(output_dir))

    if not save_crop:
        return

    bbox = detection["mask_bbox_xyxy"]
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    alpha = (mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    cutout = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = alpha
    crop_path = output_dir / "crops" / f"bubble_{index:03d}.png"
    write_image(crop_path, cutout)
    detection["crop_path"] = str(crop_path.relative_to(output_dir))


def segment_image(args) -> Path:
    if cv2 is None or np is None:
        raise RuntimeError("缺少依赖，请先安装: python -m pip install numpy opencv-python")
    if YOLO is None:
        raise RuntimeError("缺少 ultralytics，请先安装: python -m pip install ultralytics")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf 必须在 0 到 1 之间")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou 必须在 0 到 1 之间")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold 必须在 0 到 1 之间")
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0")

    image_path = args.image.expanduser().resolve()
    model_path = choose_model(args.model)
    output_dir = (args.output or image_path.with_name(f"{image_path.stem}_segmented")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)
    height, width = image.shape[:2]
    allowed_classes = parse_class_ids(args.classes)

    ensure_custom_addmodules()
    model = YOLO(str(model_path))
    if getattr(model, "task", None) not in (None, "segment"):
        raise ValueError(f"模型任务是 {getattr(model, 'task', None)!r}，需要实例分割模型。")

    predict_kwargs = {
        "source": image,
        "task": "segment",
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "retina_masks": True,
        "verbose": False,
    }
    if args.device.lower() != "auto":
        predict_kwargs["device"] = args.device
    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError("模型没有返回推理结果。")

    result = results[0]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None or len(boxes) == 0:
        detections = []
    else:
        boxes_xyxy = as_numpy(boxes.xyxy)
        confidences = as_numpy(boxes.conf).reshape(-1)
        class_ids = as_numpy(boxes.cls).reshape(-1).astype(int)
        raw_masks = as_numpy(masks.data)
        detections = []
        for index, raw_mask in enumerate(raw_masks):
            if index >= len(boxes_xyxy):
                break
            class_id = int(class_ids[index]) if index < len(class_ids) else 0
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            mask = resize_mask(raw_mask, height, width, args.mask_threshold)
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            box = clipped_box(boxes_xyxy[index], width, height)
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            detections.append(
                {
                    "mask": mask,
                    "class_id": class_id,
                    "class_name": class_name(getattr(result, "names", None), class_id),
                    "confidence": confidence,
                    "box_xyxy": box,
                    "mask_bbox_xyxy": bbox,
                    "area_px": int(mask.sum()),
                }
            )

    for index, detection in enumerate(detections, start=1):
        save_detection_outputs(image, detection, output_dir, index, not args.no_crops)

    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        combined_mask = np.maximum(combined_mask, detection["mask"].astype(np.uint8))
    write_image(output_dir / "combined_mask.png", combined_mask * 255)

    if not args.no_overlay:
        write_image(output_dir / "overlay.png", draw_overlay(image, detections))

    serializable = []
    for detection in detections:
        item = {key: value for key, value in detection.items() if key != "mask"}
        serializable.append(item)
    summary = {
        "image": str(image_path),
        "model": str(model_path),
        "image_width": width,
        "image_height": height,
        "bubble_count": len(detections),
        "combined_area_px": int(combined_mask.sum()),
        "parameters": {
            "confidence": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "mask_threshold": args.mask_threshold,
            "classes": sorted(allowed_classes) if allowed_classes is not None else None,
        },
        "detections": serializable,
    }
    if not args.no_json:
        (output_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"完成：检测到 {len(detections)} 个气泡")
    print(f"输出目录：{output_dir}")
    return output_dir


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        segment_image(args)
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
