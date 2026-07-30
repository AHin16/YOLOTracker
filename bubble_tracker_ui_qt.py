#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os.path as osp
import threading
import traceback

import cv2
import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, QEvent, QFile
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QTabWidget,
)
from PySide6.QtUiTools import QUiLoader

from bubble_analyzer.gas_holdup_qt import GasHoldupWidget
from demo_yolo11 import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    Args,
    main as run_tracker,
)
from ui_formatters import format_processing_fps

ROOT_DIR = osp.abspath(osp.dirname(__file__))
UI_STATE_PATH = osp.join(ROOT_DIR, ".bubble_tracker_ui_state.json")
INPUT_FILETYPES = (
    "Supported inputs (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.webp *.bmp *.png *.tif *.tiff);;"
    "Video files (*.mp4 *.avi *.mov *.mkv);;"
    "Image files (*.jpg *.jpeg *.webp *.bmp *.png *.tif *.tiff);;"
    "All files (*.*)"
)

# Map widget objectNames to the JSON state keys (backward-compatible with tkinter version)
_WIDGET_TO_STATE_KEY = {
    "modelEdit": "model_path",
    "inputEdit": "input_path",
    "outputDirEdit": "output_dir",
    "outputVideoEdit": "output_video",
    "outputCsvEdit": "output_csv",
    "trackerCombo": "tracker_type",
    "deviceCombo": "device",
    "confidenceSpin": "predict_conf",
    "nmsIouSpin": "predict_iou",
    "maskMatchIouSpin": "mask_match_iou",
    "matchThreshSpin": "match_thresh",
    "inputWidthSpin": "input_width",
    "inputHeightSpin": "input_height",
    "fpsSpin": "frame_rate",
    "pixelScaleSpin": "pixel_scale",
    "distanceUnitEdit": "distance_unit",
    "speedGapSpin": "speed_gap",
    "minTrackFramesSpin": "min_track_frames",
    "maxTurnAngleSpin": "max_turn_angle",
    "edgeMarginSpin": "edge_exclusion_margin",
    "trailLengthSpin": "trail_length",
    "previewStrideSpin": "preview_stride",
    "maxHistoryGapSpin": "max_history_gap",
    "showSpeedTextCheck": "show_speed_text",
}
_STATE_KEY_TO_WIDGET = {v: k for k, v in _WIDGET_TO_STATE_KEY.items()}


class TrackerWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._stop_event = threading.Event()

    def run(self):
        try:
            summary = run_tracker(
                self.args,
                progress_callback=self._on_progress,
                stop_event=self._stop_event,
            )
            self.finished.emit(summary)
        except Exception:
            self.error.emit(traceback.format_exc())

    def stop(self):
        self._stop_event.set()

    def _on_progress(self, payload):
        self.progress.emit(payload)


class BubbleTrackerQt(QObject):
    """Application controller that loads bubble_tracker_ui.ui and wires logic.

    The .ui file can be opened and edited visually in Qt Designer.
    Changes take effect at runtime without recompilation.
    """

    def __init__(self):
        super().__init__()

        loader = QUiLoader()
        ui_file = QFile(osp.join(ROOT_DIR, "bubble_tracker_ui.ui"))
        ui_file.open(QFile.ReadOnly)
        self.window = loader.load(ui_file)
        ui_file.close()

        tracking_page = self.window.takeCentralWidget()
        self.workflow_tabs = QTabWidget(self.window)
        self.workflow_tabs.setObjectName("workflowTabs")
        self.gas_holdup_widget = GasHoldupWidget()
        self.workflow_tabs.addTab(tracking_page, "Bubble Tracking")
        self.workflow_tabs.addTab(self.gas_holdup_widget, "Gas Holdup")
        self.window.setCentralWidget(self.workflow_tabs)
        self.window.setWindowTitle("Bubble Analyzer")

        self.worker_thread = None
        self.worker = None
        self.preview_pixmap = None
        self._pending_close = False

        self.window.installEventFilter(self)

        self._connect_signals()
        self._load_persisted_state()
        self._ensure_output_paths()
        self._update_form_auto_outputs()
        self.gas_holdup_widget.ensure_defaults(
            self.window.modelEdit.text().strip(),
            self.window.outputDirEdit.text().strip() or DEFAULT_OUTPUT_DIR,
        )

        self.window.show()

    # ---- Event filter (for close event) --------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.window and event.type() == QEvent.Close:
            self._save_persisted_state()
            tracker_running = self.worker_thread is not None and self.worker_thread.isRunning()
            gas_holdup_running = self.gas_holdup_widget.is_running()
            if tracker_running or gas_holdup_running:
                if self._pending_close:
                    event.ignore()
                    return True
                reply = QMessageBox.question(
                    self.window,
                    "Quit",
                    "Processing is still running. Stop and quit?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return True
                self._pending_close = True
                if tracker_running:
                    self._stop_processing()
                if gas_holdup_running:
                    self.gas_holdup_widget.stop_processing()
                event.ignore()
                return True
            event.accept()
            return True
        return False

    # ---- Signal wiring -------------------------------------------------------

    def _connect_signals(self):
        w = self.window

        # File panel browse buttons
        w.modelBrowseBtn.clicked.connect(self._browse_model)
        w.inputFileBtn.clicked.connect(self._browse_input_file)
        w.inputFolderBtn.clicked.connect(self._browse_input_dir)
        w.outputDirBrowseBtn.clicked.connect(self._browse_output_dir)
        w.outputVideoSaveBtn.clicked.connect(self._browse_output_video)
        w.outputCsvSaveBtn.clicked.connect(self._browse_output_csv)

        # Action buttons
        w.startBtn.clicked.connect(self._start_processing)
        w.stopBtn.clicked.connect(self._stop_processing)

        # Auto-refresh output paths when input or output dir changes
        w.inputEdit.textChanged.connect(self._update_form_auto_outputs)
        w.outputDirEdit.textChanged.connect(self._update_form_auto_outputs)

        # Auto-detect FPS from video when input file changes
        w.inputEdit.textChanged.connect(self._on_input_changed)

        # State persistence on any parameter change
        for widget in self._state_widgets():
            sig = self._change_signal(widget)
            if sig is not None:
                sig.connect(self._save_persisted_state)
        self.gas_holdup_widget.state_changed.connect(self._save_persisted_state)
        self.gas_holdup_widget.processing_finished.connect(self._close_when_workers_finish)

    # ---- State persistence ---------------------------------------------------

    def _state_widgets(self):
        return [getattr(self.window, attr, None) for attr in _WIDGET_TO_STATE_KEY]

    @staticmethod
    def _change_signal(widget):
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.valueChanged
        if isinstance(widget, QLineEdit):
            return widget.textChanged
        if isinstance(widget, QComboBox):
            return widget.currentTextChanged
        if isinstance(widget, QCheckBox):
            return widget.toggled
        return None

    @staticmethod
    def _widget_value(widget):
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return None

    @staticmethod
    def _set_widget_value(widget, value):
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))

    def _serialize_state(self):
        state = {}
        for widget in self._state_widgets():
            obj_name = widget.objectName()
            key = _WIDGET_TO_STATE_KEY.get(obj_name, obj_name)
            state[key] = self._widget_value(widget)
        state["gas_holdup"] = self.gas_holdup_widget.serialize_state()
        return state

    def _save_persisted_state(self, *_args):
        try:
            with open(UI_STATE_PATH, "w", encoding="utf-8") as file:
                json.dump(self._serialize_state(), file, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _load_persisted_state(self):
        if not osp.isfile(UI_STATE_PATH):
            return
        try:
            with open(UI_STATE_PATH, "r", encoding="utf-8") as file:
                state = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(state, dict):
            return

        gas_holdup_state = state.get("gas_holdup", {})
        for key, value in state.items():
            if key == "gas_holdup":
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            widget_name = _STATE_KEY_TO_WIDGET.get(key, key)
            widget = getattr(self.window, widget_name, None)
            if widget is not None:
                self._set_widget_value(widget, value)
        self.gas_holdup_widget.restore_state(gas_holdup_state)

    # ---- File panel browse handlers ------------------------------------------

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open model",
            "",
            "PyTorch/ONNX model (*.pt *.onnx);;All files (*.*)",
        )
        if file_path:
            self.window.modelEdit.setText(file_path)

    def _browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Open input", "", INPUT_FILETYPES
        )
        if file_path:
            self.window.inputEdit.setText(file_path)
            self._update_form_auto_outputs()
            self._load_video_fps(file_path)

    def _browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self.window, "Open image folder")
        if dir_path:
            self.window.inputEdit.setText(dir_path)
            self._update_form_auto_outputs()

    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self.window, "Open output folder")
        if dir_path:
            self.window.outputDirEdit.setText(dir_path)
            self._update_form_auto_outputs()

    def _browse_output_video(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.window, "Save output video", "", "MP4 video (*.mp4);;All files (*.*)"
        )
        if file_path:
            self.window.outputVideoEdit.setText(file_path)

    def _browse_output_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.window, "Save output CSV", "", "CSV file (*.csv);;All files (*.*)"
        )
        if file_path:
            self.window.outputCsvEdit.setText(file_path)

    # ---- Helpers -------------------------------------------------------------

    def _load_video_fps(self, file_path):
        cap = cv2.VideoCapture(file_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.window.fpsSpin.setValue(round(float(fps), 3))
        finally:
            cap.release()

    def _ensure_output_paths(self):
        if not self.window.outputVideoEdit.text().strip() or not self.window.outputCsvEdit.text().strip():
            self._update_form_auto_outputs()

    def _on_input_changed(self, text):
        if text and osp.isfile(text):
            self._load_video_fps(text)

    def _update_form_auto_outputs(self):
        input_path = self.window.inputEdit.text().strip()
        output_dir = self.window.outputDirEdit.text().strip() or DEFAULT_OUTPUT_DIR
        base_name = "bubble_tracking"
        if input_path:
            if osp.isdir(input_path):
                base_name = osp.basename(osp.normpath(input_path)) or base_name
            else:
                base_name = osp.splitext(osp.basename(input_path))[0] or base_name

        self.window.outputVideoEdit.blockSignals(True)
        self.window.outputCsvEdit.blockSignals(True)
        self.window.outputVideoEdit.setText(osp.join(output_dir, f"{base_name}_result.mp4"))
        self.window.outputCsvEdit.setText(osp.join(output_dir, f"{base_name}_motion.csv"))
        self.window.outputVideoEdit.blockSignals(False)
        self.window.outputCsvEdit.blockSignals(False)

    # ---- Build args from UI --------------------------------------------------

    def _build_args(self):
        w = self.window

        model_path = w.modelEdit.text().strip()
        input_path = w.inputEdit.text().strip()
        output_dir = w.outputDirEdit.text().strip() or DEFAULT_OUTPUT_DIR
        output_video = w.outputVideoEdit.text().strip()
        output_csv = w.outputCsvEdit.text().strip()

        if not model_path or not osp.isfile(model_path):
            raise ValueError("Model file does not exist.")
        if not input_path or (not osp.isfile(input_path) and not osp.isdir(input_path)):
            raise ValueError("Input path does not exist.")

        args = Args()
        args.model_path = model_path
        args.path = input_path
        args.save_result = output_dir
        args.output_video_path = output_video
        args.output_speed_csv_path = output_csv
        args.output_track_txt_path = osp.join(
            output_dir,
            f"{osp.splitext(osp.basename(output_video))[0] or 'bubble_tracking'}_tracks.txt",
        )
        args.tracker_type = w.trackerCombo.currentText().strip()
        args.device = w.deviceCombo.currentText().strip() or "auto"
        args.predict_conf = float(w.confidenceSpin.value())
        args.predict_iou = float(w.nmsIouSpin.value())
        args.mask_match_iou = float(w.maskMatchIouSpin.value())
        args.match_thresh = float(w.matchThreshSpin.value())
        args.input_size = (int(w.inputHeightSpin.value()), int(w.inputWidthSpin.value()))

        frame_rate = float(w.fpsSpin.value())
        if frame_rate <= 0:
            raise ValueError("FPS must be greater than 0.")
        min_track_frames = int(w.minTrackFramesSpin.value())
        if min_track_frames < 1:
            raise ValueError("Min Track Frames must be at least 1.")
        max_turn_angle = float(w.maxTurnAngleSpin.value())
        if max_turn_angle <= 0 or max_turn_angle > 180:
            raise ValueError("Max Turn Angle must be in the range (0, 180].")

        args.fps = frame_rate
        args.frame_interval_seconds = 1.0 / frame_rate
        args.pixel_to_real_scale = float(w.pixelScaleSpin.value())
        args.distance_unit = w.distanceUnitEdit.text().strip() or "unit"
        args.speed_frame_gap = int(w.speedGapSpin.value())
        args.min_track_frames = min_track_frames
        args.max_turn_angle_deg = max_turn_angle
        args.edge_exclusion_margin_px = float(w.edgeMarginSpin.value())
        args.trail_length = int(w.trailLengthSpin.value())
        args.max_history_gap = int(w.maxHistoryGapSpin.value())
        args.preview_stride = int(w.previewStrideSpin.value())
        args.show_speed_text = bool(w.showSpeedTextCheck.isChecked())
        return args

    # ---- Processing start / stop ---------------------------------------------

    def _start_processing(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return
        try:
            args = self._build_args()
        except Exception as exc:
            QMessageBox.critical(self.window, "Invalid parameters", str(exc))
            return

        self._set_running(True)
        self._append_log("Processing started.")

        self.worker = TrackerWorker(args)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)

        self.worker.progress.connect(self._handle_progress)
        self.worker.finished.connect(self._handle_done)
        self.worker.error.connect(self._handle_error)

        self.worker_thread.start()

    def _stop_processing(self):
        if self.worker is not None:
            self.worker.stop()
            self.window.statusLabel.setText("Stopping...")
            self._append_log("Stop requested. Waiting for current frame to finish.")

    def _on_thread_finished(self):
        self.worker_thread = None
        self._close_when_workers_finish()

    def _close_when_workers_finish(self):
        tracker_running = self.worker_thread is not None and self.worker_thread.isRunning()
        if self._pending_close and not tracker_running and not self.gas_holdup_widget.is_running():
            self._pending_close = False
            self.window.close()

    def _set_running(self, running):
        if running:
            self.window.startBtn.setEnabled(False)
            self.window.stopBtn.setEnabled(True)
            self.window.statusLabel.setText("Processing...")
            self.window.processingFpsLabel.setText(format_processing_fps(None))
        else:
            self.window.startBtn.setEnabled(True)
            self.window.stopBtn.setEnabled(False)
            self.window.statusLabel.setText("Ready")
            self.window.progressBar.reset()

    # ---- Worker signal handlers ----------------------------------------------

    def _handle_progress(self, payload):
        metrics = payload.get("metrics", {})
        frame_id = payload.get("frame_id", 0)
        total_frames = payload.get("total_frames")
        distance_unit = metrics.get("distance_unit", "unit")

        w = self.window
        w.frameValue.setText(str(frame_id))
        w.trackedValue.setText(str(metrics.get("tracked_count", 0)))
        w.avgSpeedValue.setText(f"{metrics.get('avg_speed', 0.0):.2f} {distance_unit}/s")
        w.maxSpeedValue.setText(f"{metrics.get('max_speed', 0.0):.2f} {distance_unit}/s")
        w.avgAreaValue.setText(f"{metrics.get('avg_area_px', 0.0):.2f} px^2")
        w.uniqueTracksValue.setText(str(metrics.get("unique_tracks", 0)))
        w.csvSamplesValue.setText(str(metrics.get("samples_written", 0)))
        w.processingFpsLabel.setText(format_processing_fps(metrics.get("fps")))

        if total_frames:
            w.progressBar.setMaximum(total_frames)
            w.progressBar.setValue(frame_id)
            w.progressLabel.setText(f"{frame_id}/{total_frames}")
        else:
            w.progressBar.setMaximum(max(frame_id, 1))
            w.progressBar.setValue(frame_id)
            w.progressLabel.setText(f"{frame_id} frames")

        frame = payload.get("preview_frame")
        if frame is not None:
            self._update_preview(frame)

    def _handle_done(self, summary):
        self._set_running(False)
        self.worker = None

        stopped = summary.get("stopped", False)
        status_text = "Stopped." if stopped else "Completed."
        self.window.statusLabel.setText(status_text)
        self.window.progressLabel.setText(f"{summary.get('processed_frames', 0)} frames")
        self._append_log(status_text)
        self._append_log(f"Output video: {summary.get('video_path', '')}")
        self._append_log(f"Motion CSV: {summary.get('motion_csv_path', '')}")
        QMessageBox.information(
            self.window,
            "Processing finished",
            f"{status_text}\n\n"
            f"Output video:\n{summary.get('video_path', '')}\n\n"
            f"Motion CSV:\n{summary.get('motion_csv_path', '')}",
        )

    def _handle_error(self, error_message):
        self._set_running(False)
        self.worker = None
        self.window.statusLabel.setText("Error")
        self._append_log("Error:")
        self._append_log(error_message)
        QMessageBox.critical(self.window, "Error", error_message)

    # ---- Preview -------------------------------------------------------------

    def _update_preview(self, frame_bgr):
        label = self.window.previewLabel
        max_w = max(label.width() - 8, 320)
        max_h = max(label.height() - 8, 240)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape

        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w, ch = frame_rgb.shape

        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.preview_pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(self.preview_pixmap)

    # ---- Logging -------------------------------------------------------------

    def _append_log(self, message):
        self.window.logsText.append(message)


def main():
    app = QApplication.instance() or QApplication([])
    ui = BubbleTrackerQt()
    return app.exec()


if __name__ == "__main__":
    main()
