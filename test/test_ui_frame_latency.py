#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from demo_yolo11 import process_frame


class _NoDetectionPredictor:
    def predict(self, frame, timer):
        return None, {"raw_img": frame}


class _FrameTimer:
    def __init__(self, frame_seconds):
        self.frame_seconds = frame_seconds
        self.diff = 0.0
        self.average_time = 0.0

    def stop(self, average=True):
        self.diff = self.frame_seconds
        self.average_time = self.frame_seconds
        return self.average_time if average else self.diff


class FrameLatencyMetricsTest(unittest.TestCase):
    @patch("demo_yolo11.render_mask_frame", return_value="preview")
    def test_reports_current_frame_latency_in_milliseconds(self, _render_mask_frame):
        timer = _FrameTimer(0.012345)

        _, _, metrics = process_frame(
            frame=object(),
            predictor=_NoDetectionPredictor(),
            tracker=None,
            timer=timer,
            args=SimpleNamespace(distance_unit="mm"),
            frame_id=0,
            results=[],
        )

        self.assertAlmostEqual(metrics["frame_latency_ms"], 12.345)


if __name__ == "__main__":
    unittest.main()
