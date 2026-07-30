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
            state = controller._serialize_state()

            self.assertIn("gas_holdup", state)
            self.assertEqual(state["gas_holdup"]["confidence_threshold"], 0.25)
            self.assertIn("tracker_type", state)
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
                }
            )
            gas_state = controller._serialize_state()["gas_holdup"]

            self.assertEqual(gas_state["confidence_threshold"], 0.35)
            self.assertEqual(gas_state["minimum_bubble_area"], 25)
            self.assertEqual(gas_state["roi_width"], 300)
            self.assertEqual(gas_state["edge_bubble_policy"], "exclude")
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
