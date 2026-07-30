#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bubble_analyzer.image_io import get_image_list
from bubble_analyzer.inference import create_predictor


class SharedImageIoTest(unittest.TestCase):
    def test_lists_supported_images_in_natural_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("frame10.png", "frame2.tif", "frame1.jpg", "notes.txt"):
                (root / name).touch()

            images = [Path(path).name for path in get_image_list(temp_dir)]

        self.assertEqual(images, ["frame1.jpg", "frame2.tif", "frame10.png"])


class SharedPredictorFactoryTest(unittest.TestCase):
    def test_builds_yolo_predictor_from_workflow_settings(self):
        captured = {}
        fake_module = types.ModuleType("detector_head.yolo11.predictor_yolo11")

        class FakePredictor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module.PredictorYolo11 = FakePredictor
        args = SimpleNamespace(
            model_path="bubble.pt",
            input_size=(512, 768),
            predict_conf=0.2,
            predict_iou=0.4,
            device="cpu",
        )

        with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            predictor = create_predictor(args)

        self.assertIsInstance(predictor, FakePredictor)
        self.assertEqual(
            captured,
            {
                "model_path": "bubble.pt",
                "input_size": (512, 768),
                "conf_threshold": 0.2,
                "iou_threshold": 0.4,
                "device": "cpu",
            },
        )


if __name__ == "__main__":
    unittest.main()
