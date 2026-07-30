#!/usr/bin/env python
# -*- coding: utf-8 -*-


def create_predictor(settings):
    """Create the shared YOLO predictor from workflow-compatible settings."""
    from detector_head.yolo11.predictor_yolo11 import PredictorYolo11

    return PredictorYolo11(
        model_path=settings.model_path,
        input_size=settings.input_size,
        conf_threshold=getattr(settings, "predict_conf", 0.1),
        iou_threshold=getattr(settings, "predict_iou", 0.45),
        device=getattr(settings, "device", "auto"),
    )

