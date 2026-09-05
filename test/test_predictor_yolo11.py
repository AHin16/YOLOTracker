#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from detector_head.yolo11.predictor_yolo11 import PredictorYolo11


def _fake_model(mask):
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.xyxy = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    boxes.data = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.9, 0.0]])
    result = SimpleNamespace(
        boxes=boxes,
        masks=SimpleNamespace(data=torch.tensor(mask[None]), xy=[]),
    )
    model = MagicMock()
    model.task = "segment"
    model.predict.return_value = [result]
    return model


class PredictorYolo11QualityTest(unittest.TestCase):
    def test_keeps_legacy_predict_arguments_when_quality_is_disabled(self):
        model = _fake_model(np.ones((2, 2), dtype=np.float32))
        with patch("detector_head.yolo11.predictor_yolo11.YOLO", return_value=model):
            predictor = PredictorYolo11("unused.pt", high_quality_segmentation=False)
            predictor.predict(np.zeros((4, 6, 3), dtype=np.uint8))

        kwargs = model.predict.call_args.kwargs
        self.assertNotIn("imgsz", kwargs)
        self.assertNotIn("retina_masks", kwargs)

    def test_enables_retina_masks_and_resizes_before_thresholding(self):
        source_mask = np.asarray([[0.9, 0.6], [0.4, 0.1]], dtype=np.float32)
        model = _fake_model(source_mask)
        with patch("detector_head.yolo11.predictor_yolo11.YOLO", return_value=model):
            predictor = PredictorYolo11(
                "unused.pt",
                input_size=(4, 6),
                high_quality_segmentation=True,
            )
            _, image_info = predictor.predict(np.zeros((4, 6, 3), dtype=np.uint8))

        kwargs = model.predict.call_args.kwargs
        self.assertEqual(kwargs["imgsz"], (4, 6))
        self.assertTrue(kwargs["retina_masks"])
        expected = cv2.resize(source_mask, (6, 4), interpolation=cv2.INTER_LINEAR) > 0.5
        np.testing.assert_array_equal(image_info["mask_arrays"][0], expected.astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
