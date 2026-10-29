import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox

import bubble_tracker_ui_qt


class UiLanguageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state_path = Path(temporary.name) / "state.json"
        state_patch = patch.object(bubble_tracker_ui_qt, "UI_STATE_PATH", str(self.state_path))
        state_patch.start()
        self.addCleanup(state_patch.stop)

    def controller(self):
        with patch.object(QMainWindow, "show"):
            controller = bubble_tracker_ui_qt.BubbleTrackerQt()
        self.addCleanup(controller.window.deleteLater)
        self.addCleanup(controller.window.close)
        return controller

    def test_switches_all_workflows_without_changing_inputs(self):
        controller = self.controller()
        self.assertEqual(controller.workflow_tabs.tabText(0), "气泡跟踪")
        self.assertEqual(controller.workflow_tabs.tabText(1), "气含率")
        controller.window.inputEdit.setText("C:/实验/Ready.png")
        controller.window.fpsSpin.setValue(172)
        controller.window.deviceCombo.setCurrentIndex(2)
        gas = controller.gas_holdup_widget
        gas.edge_policy_combo.setCurrentIndex(1)
        gas.high_quality_check.setChecked(True)
        before = controller._serialize_state()
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("en"))
        self.assertEqual(controller.window.startBtn.text(), "Start")
        self.assertEqual(controller.workflow_tabs.tabText(2), "Flow State Warning")
        self.assertEqual(gas.start_button.text(), "Start Gas Holdup")
        self.assertEqual(controller.flow_warning_widget.start_button.text(), "Start Warning Analysis")
        after = controller._serialize_state()
        self.assertEqual(after.pop("language"), "en")
        before.pop("language")
        self.assertEqual(before, after)
        self.assertEqual(after["device"], "cuda:0")
        self.assertEqual(after["gas_holdup"]["edge_bubble_policy"], "exclude")
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for label in controller.window.findChildren(QLabel)
                             for char in label.text()))
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for widget in controller.window.findChildren(QLabel)
                             for char in widget.toolTip()))
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("zh_CN"))
        self.assertEqual(controller.window.startBtn.text(), "开始")
        self.assertEqual(controller.window.fpsSpin.value(), 172)

    def test_translated_options_keep_backend_configuration_values(self):
        controller = self.controller()
        model_path = self.state_path.parent / "model.pt"
        model_path.touch()
        input_folder = self.state_path.parent / "images"
        input_folder.mkdir()
        controller.window.modelEdit.setText(str(model_path))
        controller.window.inputEdit.setText(str(input_folder))
        controller.window.trackerCombo.setCurrentIndex(1)
        controller.window.deviceCombo.setCurrentIndex(2)
        gas = controller.gas_holdup_widget
        gas.model_edit.setText(str(model_path))
        gas.input_folder_edit.setText(str(input_folder))
        gas.edge_policy_combo.setCurrentIndex(2)
        for language in ("zh_CN", "en", "zh_CN"):
            controller.language.set_language(language)
            args = controller._build_args()
            self.assertEqual(args.tracker_type, "bytetrack")
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(gas._build_config().edge_policy.value, "weighted")
            self.assertEqual(controller.flow_warning_widget.serialize_state()["device"], "auto")

    def test_remembers_language_and_supports_old_or_invalid_settings(self):
        self.state_path.write_text(json.dumps({"frame_rate": 172, "language": "en"}), encoding="utf-8")
        controller = self.controller()
        self.assertEqual(controller.workflow_tabs.tabText(0), "Bubble Tracking")
        self.assertEqual(controller.window.fpsSpin.value(), 172)
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("zh_CN"))
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["language"], "zh_CN")
        controller.window.close()
        self.state_path.write_text(json.dumps({"frame_rate": 100, "language": "unknown"}), encoding="utf-8")
        restored = self.controller()
        self.assertEqual(restored.workflow_tabs.tabText(0), "气泡跟踪")
        self.assertEqual(restored.window.fpsSpin.value(), 100)

    def test_switches_processing_text_and_preserves_preview(self):
        import numpy as np

        controller = self.controller()
        controller._set_running(True)
        controller._handle_progress({"frame_id": 3, "metrics": {"fps": 12.5, "frame_latency_ms": 80},
                                     "preview_frame": np.zeros((32, 32, 3), dtype=np.uint8)})
        controller._append_log("Processing started.")
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("en"))
        self.assertEqual(controller.window.statusLabel.text(), "Processing...")
        self.assertEqual(controller.window.progressLabel.text(), "3 frames")
        self.assertEqual(controller.window.processingFpsLabel.text(), "Processing FPS: 12.50")
        self.assertIn("Processing started.", controller.window.logsText.toPlainText())
        self.assertFalse(controller.window.previewLabel.pixmap().isNull())
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("zh_CN"))
        self.assertEqual(controller.window.statusLabel.text(), "正在处理…")
        self.assertEqual(controller.window.progressLabel.text(), "3 帧")
        self.assertFalse(controller.window.previewLabel.pixmap().isNull())
        self.assertFalse(controller.window.startBtn.isEnabled())

    def test_translates_warning_reasons_events_and_plots_without_reset(self):
        controller = self.controller()
        widget = controller.flow_warning_widget
        reason = "面积占比 20 高于正常界限 10"
        event = {"group": "appearance", "start_time": 0, "trigger_time": 0.5,
                 "end_time": None, "end_reason": None, "reason": reason, "evidence_frame": 5}
        widget._handle_progress({
            "phase": "analysis", "frame_id": 5, "total_frames": 10, "events": [event],
            "window": {"state": "warning", "start_time": 0, "end_time": 0.5,
                       "metrics": {"area_ratio": 20, "bubble_count": 5},
                       "groups": {"appearance": {"state": "warning", "reason": reason}}},
        })
        count = len(widget.plots["appearance"].values)
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("en"))
        self.assertIn("Warning", widget.overall_label.text())
        self.assertIn("above", widget.reason_labels["appearance"].text())
        self.assertIn("above", widget.event_table.item(0, 3).text())
        self.assertEqual(widget.event_table.item(0, 0).text(), "Bubble Appearance")
        self.assertEqual(widget.event_table.item(0, 0).data(256), 5)
        self.assertEqual(len(widget.plots["appearance"].values), count)
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("zh_CN"))
        self.assertEqual(widget.reason_labels["appearance"].text(), reason)
        reasons = reason + "；向上速度中位数 -5 低于正常界限 -1"
        controller.language.set_language("en")
        self.assertEqual(controller.language.text(reasons),
                         "Area Ratio 20 above normal limit 10; Median Upward Velocity -5 below normal limit -1")

    def test_validation_and_completion_dialogs_follow_current_language(self):
        controller = self.controller()
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("en"))
        with patch.object(QMessageBox, "critical") as critical:
            controller.flow_warning_widget.start_processing()
        self.assertEqual(critical.call_args.args[1], "Check Warning Parameters")
        self.assertIn("acquisition", critical.call_args.args[2])
        controller.language_combo.setCurrentIndex(controller.language_combo.findData("zh_CN"))
        with patch.object(QMessageBox, "information") as information:
            controller._handle_done({"processed_frames": 3, "video_path": "C:/结果/Ready.mp4"})
        self.assertEqual(information.call_args.args[1], "处理结束")
        self.assertIn("已完成", information.call_args.args[2])
        self.assertIn("C:/结果/Ready.mp4", information.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
