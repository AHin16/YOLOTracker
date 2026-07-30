#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from bubble_analyzer.gas_holdup import EdgeBubblePolicy, ROI
from bubble_analyzer.gas_holdup_workflow import GasHoldupConfig, run_gas_holdup


class FakeSegmentationPredictor:
    task = "segment"

    def predict(self, image):
        height, width = image.shape[:2]
        first = np.zeros((height, width), dtype=np.uint8)
        second = np.zeros((height, width), dtype=np.uint8)
        first[1:4, 1:4] = 1
        second[5:8, 5:9] = 1
        return None, {"mask_arrays": [first, second], "mask_polygons": []}


class GasHoldupWorkflowTest(unittest.TestCase):
    def test_converts_pixel_diameter_to_calibrated_units(self):
        config = GasHoldupConfig(
            model_path="unused.pt",
            input_folder="input",
            output_overlay_folder="overlay",
            output_csv="results.csv",
            calibration_pixels=100.0,
            calibration_length=1.0,
            distance_unit="mm",
        )

        self.assertAlmostEqual(config.real_units_per_pixel, 0.01)

    def test_rejects_invalid_calibration_values_and_units(self):
        base_config = {
            "model_path": "unused.pt",
            "input_folder": "input",
            "output_overlay_folder": "overlay",
            "output_csv": "results.csv",
        }

        with self.assertRaisesRegex(ValueError, "Calibration Pixels"):
            GasHoldupConfig(**base_config, calibration_pixels=0).real_units_per_pixel
        with self.assertRaisesRegex(ValueError, "Calibration Length"):
            GasHoldupConfig(**base_config, calibration_length=math.nan).real_units_per_pixel
        with self.assertRaisesRegex(ValueError, "Distance Unit"):
            GasHoldupConfig(**base_config, distance_unit="=1+1").normalized_distance_unit

    def test_rejects_overlay_folder_inside_input_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            cv2.imwrite(str(input_dir / "frame.png"), np.full((8, 8, 3), 100, np.uint8))

            with self.assertRaisesRegex(ValueError, "outside the Input Folder"):
                run_gas_holdup(
                    GasHoldupConfig(
                        model_path="unused.pt",
                        input_folder=str(input_dir),
                        output_overlay_folder=str(input_dir / "overlays"),
                        output_csv=str(Path(temp_dir) / "results.csv"),
                    ),
                    predictor_factory=lambda _config: FakeSegmentationPredictor(),
                )

    def test_processes_each_image_to_overlay_csv_and_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            overlay_dir = root / "overlay"
            csv_path = root / "results.csv"
            input_dir.mkdir()
            cv2.imwrite(str(input_dir / "frame2.png"), np.full((10, 10, 3), 80, np.uint8))
            cv2.imwrite(str(input_dir / "frame1.png"), np.full((10, 10, 3), 120, np.uint8))
            progress = []
            config = GasHoldupConfig(
                model_path="unused.pt",
                input_folder=str(input_dir),
                output_overlay_folder=str(overlay_dir),
                output_csv=str(csv_path),
                confidence_threshold=0.15,
                min_bubble_area=2,
                roi=ROI(),
                edge_policy=EdgeBubblePolicy.INCLUDE,
                calibration_pixels=100.0,
                calibration_length=1.0,
                distance_unit="mm",
            )

            summary = run_gas_holdup(
                config,
                progress_callback=progress.append,
                predictor_factory=lambda _config: FakeSegmentationPredictor(),
            )

            overlays = sorted(overlay_dir.glob("*.png"))
            with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(overlays), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            list(rows[0]),
            [
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
            ],
        )
        self.assertEqual(rows[0]["Frame"], "1")
        self.assertEqual(rows[0]["BubbleCount"], "2")
        self.assertEqual(rows[0]["BubbleArea"], "21")
        self.assertEqual(rows[0]["ROIArea"], "100")
        self.assertAlmostEqual(float(rows[0]["GasHoldup"]), 21.0)
        expected_diameter_px = (
            2.0 * math.sqrt(9.0 / math.pi) + 2.0 * math.sqrt(12.0 / math.pi)
        ) / 2.0
        self.assertAlmostEqual(float(rows[0]["AverageDiameterPx"]), expected_diameter_px, places=5)
        self.assertAlmostEqual(float(rows[0]["AverageDiameter"]), expected_diameter_px * 0.01, places=5)
        self.assertEqual(rows[0]["DiameterUnit"], "mm")
        self.assertAlmostEqual(float(rows[0]["RealUnitsPerPixel"]), 0.01)
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0]["current_frame"], "frame1.png")
        self.assertAlmostEqual(progress[0]["metrics"]["average_diameter_px"], expected_diameter_px)
        self.assertAlmostEqual(progress[0]["metrics"]["average_diameter"], expected_diameter_px * 0.01)
        self.assertEqual(progress[0]["metrics"]["distance_unit"], "mm")
        self.assertEqual(progress[-1]["frame_id"], 2)
        self.assertFalse(summary["stopped"])
        self.assertEqual(summary["processed_frames"], 2)

    def test_honors_stop_event_between_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            for index in range(3):
                cv2.imwrite(
                    str(input_dir / f"frame{index}.png"),
                    np.full((8, 8, 3), 100, np.uint8),
                )
            stop_event = threading.Event()

            def stop_after_first(_payload):
                stop_event.set()

            summary = run_gas_holdup(
                GasHoldupConfig(
                    model_path="unused.pt",
                    input_folder=str(input_dir),
                    output_overlay_folder=str(root / "overlay"),
                    output_csv=str(root / "results.csv"),
                    min_bubble_area=1,
                ),
                progress_callback=stop_after_first,
                stop_event=stop_event,
                predictor_factory=lambda _config: FakeSegmentationPredictor(),
            )

        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["processed_frames"], 1)


if __name__ == "__main__":
    unittest.main()
