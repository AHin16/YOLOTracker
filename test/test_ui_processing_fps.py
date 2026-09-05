#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from ui_formatters import format_frame_latency_ms, format_processing_fps


ROOT_DIR = Path(__file__).resolve().parents[1]


class ProcessingFpsUiTest(unittest.TestCase):
    def test_formats_processing_fps_for_display(self):
        self.assertEqual(format_processing_fps(12.345), "Processing FPS: 12.35")

    def test_uses_placeholder_when_processing_fps_is_unavailable(self):
        self.assertEqual(format_processing_fps(None), "Processing FPS: --")
        self.assertEqual(format_processing_fps(math.nan), "Processing FPS: --")

    def test_formats_frame_latency_for_display(self):
        self.assertEqual(format_frame_latency_ms(12.345), "Frame Latency: 12.35 ms")

    def test_uses_placeholder_when_frame_latency_is_unavailable(self):
        self.assertEqual(format_frame_latency_ms(None), "Frame Latency: --")
        self.assertEqual(format_frame_latency_ms(math.nan), "Frame Latency: --")

    def test_places_processing_fps_beside_progress_bar(self):
        root = ET.parse(ROOT_DIR / "bubble_tracker_ui.ui").getroot()
        progress_layout = root.find(".//layout[@name='progressLayout']")

        self.assertIsNotNone(progress_layout)
        widget_names = [
            item.find("widget").get("name")
            for item in progress_layout.findall("item")
            if item.find("widget") is not None
        ]
        self.assertEqual(
            widget_names,
            ["progressBar", "processingFpsLabel", "frameLatencyLabel"],
        )


if __name__ == "__main__":
    unittest.main()
