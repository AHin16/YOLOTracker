import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

import bubble_tracker_ui_qt


class FlowWarningQtTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def controller(self, root):
        state_patch = patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", str(root / "state.json"))
        state_patch.start()
        self.addCleanup(state_patch.stop)
        with patch.object(QMainWindow, "show"):
            controller = bubble_tracker_ui_qt.BubbleTrackerQt()
        self.addCleanup(controller.window.deleteLater)
        return controller

    def test_third_tab_and_separate_configuration_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self.controller(Path(temp_dir))
            tabs = controller.workflow_tabs
            self.assertEqual(tabs.count(), 3)
            self.assertEqual(tabs.tabText(2), "流态安全预警")
            widget = controller.flow_warning_widget
            tracking_before = controller.window.fpsSpin.value()
            holdup_before = controller.gas_holdup_widget.serialize_state()
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", str(Path(temp_dir) / "state.json")):
                widget.restore_state({"fps": 172, "pixel_scale": 0.01, "baseline_path": "baseline"})
                saved = json.loads(json.dumps(controller._serialize_state()))
                widget.fps_spin.setValue(30)
                widget.restore_state(saved["flow_warning"])
            self.assertEqual(widget.fps_spin.value(), 172)
            self.assertEqual(controller.window.fpsSpin.value(), tracking_before)
            self.assertEqual(controller.gas_holdup_widget.serialize_state(), holdup_before)

    def test_missing_metrics_show_insufficient_and_zero_count_is_visible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self.controller(Path(temp_dir))
            widget = controller.flow_warning_widget
            widget._handle_progress({
                "phase": "analysis", "frame_id": 5, "total_frames": 10,
                "window": {
                    "start_time": 0, "end_time": 0.5, "state": "insufficient",
                    "metrics": {"area_ratio": 0.0, "bubble_count": 0.0},
                    "groups": {
                        "appearance": {"state": "normal", "reason": "有效空帧"},
                        "size": {"state": "insufficient", "reason": "无完整气泡"},
                        "motion": {"state": "insufficient", "reason": "无运动样本"},
                    },
                },
            })
            self.assertIn("0.00", widget.metric_labels["appearance"].text())
            self.assertIn("数据不足", widget.metric_labels["motion"].text())
            self.assertIn("数据不足", widget.overall_label.text())

    def test_closing_window_requests_warning_worker_stop_and_waits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self.controller(Path(temp_dir))
            widget = controller.flow_warning_widget
            event = QCloseEvent()
            with patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", str(Path(temp_dir) / "state.json")), \
                    patch.object(widget, "is_running", return_value=True), \
                    patch.object(widget, "stop_processing") as stop, \
                    patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                self.assertTrue(controller.eventFilter(controller.window, event))
                stop.assert_called_once()
                self.assertFalse(event.isAccepted())
                self.assertTrue(controller._pending_close)
                with patch.object(controller.window, "close") as close:
                    controller._close_when_workers_finish()
                    close.assert_not_called()
            with patch.object(widget, "is_running", return_value=False), patch.object(controller.window, "close") as close:
                controller._close_when_workers_finish()
                close.assert_called_once()

    def test_build_requires_explicit_time_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = self.controller(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "采集"):
                controller.flow_warning_widget._build_config()


if __name__ == "__main__":
    unittest.main()
