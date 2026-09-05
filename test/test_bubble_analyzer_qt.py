#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

import bubble_tracker_ui_qt


class BubbleAnalyzerQtTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_hosts_tracking_and_gas_holdup_as_independent_tabs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            tabs = controller.workflow_tabs
            self.assertEqual(tabs.count(), 2)
            self.assertEqual(tabs.tabText(0), "Bubble Tracking")
            self.assertEqual(tabs.tabText(1), "Gas Holdup")
            self.assertIsNotNone(tabs.widget(0).findChild(QPushButton, "startBtn"))
            self.assertIs(tabs.widget(1), controller.gas_holdup_widget)
            controller.window.close()

    def test_serializes_gas_holdup_settings_in_a_namespace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            controller.gas_holdup_widget.confidence_spin.setValue(0.25)
            controller.gas_holdup_widget.calibration_pixels_spin.setValue(100.0)
            controller.gas_holdup_widget.calibration_length_spin.setValue(1.0)
            controller.gas_holdup_widget.distance_unit_edit.setText("mm")
            controller.gas_holdup_widget.high_quality_check.setChecked(True)
            controller.gas_holdup_widget.input_width_spin.setValue(1280)
            controller.gas_holdup_widget.input_height_spin.setValue(720)
            state = controller._serialize_state()

            self.assertIn("gas_holdup", state)
            self.assertEqual(state["gas_holdup"]["confidence_threshold"], 0.25)
            self.assertEqual(state["gas_holdup"]["calibration_pixels"], 100.0)
            self.assertEqual(state["gas_holdup"]["calibration_length"], 1.0)
            self.assertEqual(state["gas_holdup"]["distance_unit"], "mm")
            self.assertTrue(state["gas_holdup"]["high_quality_segmentation"])
            self.assertEqual(state["gas_holdup"]["input_width"], 1280)
            self.assertEqual(state["gas_holdup"]["input_height"], 720)
            self.assertIn("tracker_type", state)
            controller.window.close()

    def test_high_quality_controls_are_opt_in(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            widget = controller.gas_holdup_widget
            self.assertFalse(widget.high_quality_check.isChecked())
            self.assertFalse(widget.input_width_spin.isEnabled())
            self.assertFalse(widget.input_height_spin.isEnabled())

            widget.high_quality_check.setChecked(True)
            widget.input_width_spin.setValue(1280)
            widget.input_height_spin.setValue(720)

            model_path = os.path.join(temp_dir, "model.pt")
            input_dir = os.path.join(temp_dir, "input")
            open(model_path, "wb").close()
            os.mkdir(input_dir)
            widget.model_edit.setText(model_path)
            widget.input_folder_edit.setText(input_dir)
            widget.overlay_folder_edit.setText(os.path.join(temp_dir, "overlay"))
            widget.output_csv_edit.setText(os.path.join(temp_dir, "results.csv"))
            config = widget._build_config()

            self.assertTrue(widget.input_width_spin.isEnabled())
            self.assertTrue(widget.input_height_spin.isEnabled())
            self.assertTrue(config.high_quality_segmentation)
            self.assertEqual(config.input_size, (720, 1280))
            controller.window.close()

    def test_displays_actual_and_pixel_diameter_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            controller.gas_holdup_widget._handle_progress(
                {
                    "frame_id": 1,
                    "total_frames": 1,
                    "current_frame": "frame.png",
                    "metrics": {
                        "average_diameter": 0.1234,
                        "average_diameter_px": 12.34,
                        "distance_unit": "mm",
                    },
                }
            )
            self.assertEqual(controller.gas_holdup_widget.average_diameter_value.text(), "0.1234 mm")
            self.assertEqual(controller.gas_holdup_widget.average_diameter_px_value.text(), "12.34 px")
            controller.window.close()

    def test_restores_gas_holdup_state_without_changing_tracking_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            controller.gas_holdup_widget.restore_state(
                {
                    "confidence_threshold": 0.35,
                    "minimum_bubble_area": 25,
                    "roi_x": 10,
                    "roi_y": 20,
                    "roi_width": 300,
                    "roi_height": 400,
                    "edge_bubble_policy": "exclude",
                    "calibration_pixels": 200.0,
                    "calibration_length": 2.0,
                    "distance_unit": "mm",
                    "high_quality_segmentation": True,
                    "input_width": 1024,
                    "input_height": 768,
                }
            )
            gas_state = controller._serialize_state()["gas_holdup"]

            self.assertEqual(gas_state["confidence_threshold"], 0.35)
            self.assertEqual(gas_state["minimum_bubble_area"], 25)
            self.assertEqual(gas_state["roi_width"], 300)
            self.assertEqual(gas_state["edge_bubble_policy"], "exclude")
            self.assertEqual(gas_state["calibration_pixels"], 200.0)
            self.assertEqual(gas_state["calibration_length"], 2.0)
            self.assertEqual(gas_state["distance_unit"], "mm")
            self.assertTrue(gas_state["high_quality_segmentation"])
            self.assertEqual(gas_state["input_width"], 1024)
            self.assertEqual(gas_state["input_height"], 768)
            controller.window.close()

    def test_legacy_tracking_state_seeds_new_workflow_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "ui-state.json")
            with open(state_path, "w", encoding="utf-8") as state_file:
                json.dump(
                    {
                        "model_path": "C:/models/bubbles.pt",
                        "output_dir": "C:/results",
                    },
                    state_file,
                )
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", state_path), patch.object(
                QMainWindow, "show"
            ):
                controller = bubble_tracker_ui_qt.BubbleTrackerQt()

            gas_state = controller.gas_holdup_widget.serialize_state()
            self.assertEqual(gas_state["model_path"], "C:/models/bubbles.pt")
            self.assertTrue(gas_state["output_overlay_folder"].startswith("C:/results"))
            self.assertTrue(gas_state["output_csv"].startswith("C:/results"))
            controller.window.close()


if __name__ == "__main__":
    unittest.main()
