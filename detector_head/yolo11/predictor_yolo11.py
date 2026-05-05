# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/27 下午8:34
@Author  : Kend
@FileName: predictor_yolo11.py
@Software: PyCharm
@modifier:
"""

import os
import os.path as osp
import sys
import types

LOCAL_YOLO_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".ultralytics"))
os.makedirs(LOCAL_YOLO_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", LOCAL_YOLO_CONFIG_DIR)

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn


def _ensure_custom_addmodules():
    addmodules_name = "ultralytics.nn.Addmodules"
    bifpn_module_name = "ultralytics.nn.Addmodules.BiFPN"
    if bifpn_module_name in sys.modules:
        return

    addmodules_pkg = types.ModuleType(addmodules_name)
    bifpn_module = types.ModuleType(bifpn_module_name)

    class swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    class Bi_FPN(nn.Module):
        def __init__(self, weight_num=2, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(int(weight_num), dtype=torch.float32), requires_grad=True)
            self.epsilon = 1e-4
            self.swish = swish()

        def forward(self, x):
            if not isinstance(x, (list, tuple)):
                return x
            if len(x) == 0:
                return x

            weight_param = self.weight
            if weight_param.numel() == len(x):
                weights = weight_param
            elif weight_param.numel() > len(x):
                weights = weight_param[: len(x)]
            else:
                pad = torch.ones(len(x) - weight_param.numel(), device=weight_param.device, dtype=weight_param.dtype)
                weights = torch.cat([weight_param, pad], dim=0)

            weights = self.swish(weights)
            weights = weights / (torch.sum(weights, dim=0) + self.epsilon)

            fused = x[0] * weights[0]
            for index in range(1, len(x)):
                fused = fused + x[index] * weights[index]
            return fused

    bifpn_module.swish = swish
    bifpn_module.Bi_FPN = Bi_FPN
    addmodules_pkg.BiFPN = bifpn_module

    sys.modules[addmodules_name] = addmodules_pkg
    sys.modules[bifpn_module_name] = bifpn_module

    try:
        import ultralytics.nn as ultralytics_nn

        setattr(ultralytics_nn, "Addmodules", addmodules_pkg)
    except Exception:
        pass


def _normalize_device(device):
    if device is None:
        return None

    device_text = str(device).strip()
    if not device_text or device_text.lower() == "auto":
        return None
    if device_text.isdigit():
        return int(device_text)
    if device_text.lower() == "gpu":
        return "cuda:0"
    return device_text


def _validate_device(device):
    if device is None:
        return

    device_text = str(device).lower()
    wants_cuda = isinstance(device, int) or device_text.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was selected, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or choose CPU/auto."
        )


class PredictorYolo11:
    def __init__(self, model_path, input_size=(640, 640), conf_threshold=0.1, iou_threshold=0.45, device="auto"):
        _ensure_custom_addmodules()
        self.device = _normalize_device(device)
        _validate_device(self.device)
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.task = "segment" if getattr(self.model, "task", None) == "segment" else "detect"

    def predict(self, image: np.ndarray, timer=None):
        img_info = {"id": 0, "file_name": None}
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image
        img_info["ratio"] = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        img_info["mask_boxes"] = np.empty((0, 4), dtype=np.float32)
        img_info["mask_polygons"] = []

        if timer is not None:
            timer.start()

        predict_kwargs = {
            "source": image,
            "task": self.task,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results_list = self.model.predict(**predict_kwargs)

        if not results_list:
            return None, img_info

        result = results_list[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None, img_info

        img_info["mask_boxes"] = result.boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
        if result.masks is not None and getattr(result.masks, "xy", None) is not None:
            img_info["mask_polygons"] = [
                np.asarray(polygon, dtype=np.float32) if len(polygon) >= 3 else None
                for polygon in result.masks.xy
            ]

        return result.boxes.data, img_info


if __name__ == "__main__":
    image = r"D:\kend\myPython\ultralytics-main\ultralytics\assets\bus.jpg"
    model_path = r"D:\kend\myPython\Hk_Tracker\detector_head\yolo11\yolo11s.onnx"
    img = cv2.imread(image)
    predictor_yolo11 = PredictorYolo11(model_path)
    resource, info = predictor_yolo11.predict(img)
    print(resource, info)
