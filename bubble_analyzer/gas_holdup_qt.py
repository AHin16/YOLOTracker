#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import threading

import cv2

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .gas_holdup import EdgeBubblePolicy, ROI
from .gas_holdup_workflow import GasHoldupConfig, run_gas_holdup


class GasHoldupWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()

    def run(self):
        try:
            summary = run_gas_holdup(
                self.config,
                progress_callback=self.progress.emit,
                stop_event=self._stop_event,
            )
            self.finished.emit(summary)
        except Exception as exc:
            self.error.emit(str(exc) or exc.__class__.__name__)

    def stop(self):
        self._stop_event.set()


class PathField(QWidget):
    def __init__(self, button_text="Browse...", parent=None):
        super().__init__(parent)
        self.edit = QLineEdit(self)
        self.button = QPushButton(button_text, self)
        self.button.setMinimumWidth(92)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)


class GasHoldupWidget(QWidget):
    state_changed = Signal()
    processing_finished = Signal()

    def __init__(self, default_model_path="", default_output_dir="", parent=None):
        super().__init__(parent)
        self.worker_thread = None
        self.worker = None
        self.preview_pixmap = None
        self._last_auto_overlay = ""
        self._last_auto_csv = ""

        self._build_ui()
        self._connect_signals()
        self.ensure_defaults(default_model_path, default_output_dir)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        subtitle = QLabel(
            "Estimate gas holdup directly from YOLO instance-segmentation masks. "
            "This workflow does not use tracking, IDs, velocity, or lifetime filters."
        )
        subtitle.setWordWrap(True)
        root_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        scroll = QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(460)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.addWidget(self._build_files_group())
        left_layout.addWidget(self._build_parameters_group())
        left_layout.addWidget(self._build_actions_group())
        left_layout.addStretch(1)
        scroll.setWidget(left_panel)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 4, 4, 4)
        right_layout.addWidget(self._build_results_group())

        preview_group = QGroupBox("Preview", right_panel)
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Segmentation overlay will appear here.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet("background: #20242a; color: #aeb6c2;")
        preview_layout.addWidget(self.preview_label, 1)
        right_layout.addWidget(preview_group, 1)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 1000])

    def _build_files_group(self):
        group = QGroupBox("Files")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.model_field = PathField("Browse...")
        self.input_field = PathField("Folder...")
        self.overlay_field = PathField("Folder...")
        self.csv_field = PathField("Save As...")
        self.model_edit = self.model_field.edit
        self.input_folder_edit = self.input_field.edit
        self.overlay_folder_edit = self.overlay_field.edit
        self.output_csv_edit = self.csv_field.edit
        self._add_form_row(form, "Model", self.model_field, self.model_edit)
        self._add_form_row(form, "Input Folder", self.input_field, self.input_folder_edit)
        self._add_form_row(form, "Output Overlay Folder", self.overlay_field, self.overlay_folder_edit)
        self._add_form_row(form, "Output CSV", self.csv_field, self.output_csv_edit)
        return group

    def _build_parameters_group(self):
        group = QGroupBox("Parameters")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.10)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000000)
        self.min_area_spin.setValue(10)
        self.calibration_pixels_spin = QDoubleSpinBox()
        self.calibration_pixels_spin.setRange(0.001, 1000000000.0)
        self.calibration_pixels_spin.setDecimals(3)
        self.calibration_pixels_spin.setValue(100.0)
        self.calibration_pixels_spin.setSuffix(" px")
        self.calibration_pixels_spin.setToolTip(
            "Pixel length corresponding to the calibrated actual length."
        )
        self.calibration_length_spin = QDoubleSpinBox()
        self.calibration_length_spin.setRange(0.000001, 1000000000.0)
        self.calibration_length_spin.setDecimals(6)
        self.calibration_length_spin.setValue(1.0)
        self.calibration_length_spin.setToolTip(
            "Actual length corresponding to Calibration Pixels."
        )
        self.distance_unit_edit = QLineEdit("mm")
        self.distance_unit_edit.setToolTip("Unit used for the calibrated bubble diameter.")
        self.edge_policy_combo = QComboBox()
        for label, policy in (
            ("Include", EdgeBubblePolicy.INCLUDE),
            ("Exclude", EdgeBubblePolicy.EXCLUDE),
            ("Weighted", EdgeBubblePolicy.WEIGHTED),
        ):
            self.edge_policy_combo.addItem(label, policy.value)

        roi_widget = QWidget()
        roi_layout = QGridLayout(roi_widget)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setHorizontalSpacing(8)
        self.roi_x_spin = self._roi_spin("ROI left coordinate")
        self.roi_y_spin = self._roi_spin("ROI top coordinate")
        self.roi_width_spin = self._roi_spin("Zero extends to the frame right edge")
        self.roi_height_spin = self._roi_spin("Zero extends to the frame bottom edge")
        for column, (label, widget) in enumerate(
            (
                ("X", self.roi_x_spin),
                ("Y", self.roi_y_spin),
                ("W", self.roi_width_spin),
                ("H", self.roi_height_spin),
            )
        ):
            roi_layout.addWidget(QLabel(label), 0, column)
            roi_layout.addWidget(widget, 1, column)

        self._add_form_row(form, "Confidence Threshold", self.confidence_spin)
        self._add_form_row(form, "Minimum Bubble Area (px)", self.min_area_spin)
        self._add_form_row(form, "Calibration Pixels", self.calibration_pixels_spin)
        self._add_form_row(form, "Actual Length", self.calibration_length_spin)
        self._add_form_row(form, "Distance Unit", self.distance_unit_edit)
        self._add_form_row(form, "ROI (0 size = frame edge)", roi_widget)
        self._add_form_row(form, "Edge Bubble Policy", self.edge_policy_combo)
        return group

    def _build_actions_group(self):
        group = QGroupBox("Processing")
        layout = QVBoxLayout(group)
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Gas Holdup")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("0/0")
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        return group

    def _build_results_group(self):
        group = QGroupBox("Results")
        form = QFormLayout(group)
        self.current_frame_value = QLabel("--")
        self.bubble_count_value = QLabel("0")
        self.bubble_area_value = QLabel("0 px")
        self.roi_area_value = QLabel("0 px")
        self.area_ratio_value = QLabel("0.00 %")
        self.gas_holdup_value = QLabel("0.00 %")
        self.average_diameter_value = QLabel("0.0000 mm")
        self.average_diameter_px_value = QLabel("0.00 px")
        self.processing_fps_value = QLabel("--")
        for label, widget in (
            ("Current Frame", self.current_frame_value),
            ("Bubble Count", self.bubble_count_value),
            ("Bubble Area", self.bubble_area_value),
            ("ROI Area", self.roi_area_value),
            ("Bubble Area Ratio", self.area_ratio_value),
            ("Gas Holdup", self.gas_holdup_value),
            ("Average Bubble Diameter", self.average_diameter_value),
            ("Average Bubble Diameter (px)", self.average_diameter_px_value),
            ("Processing FPS", self.processing_fps_value),
        ):
            form.addRow(label, widget)
        return group

    @staticmethod
    def _add_form_row(form, text, widget, buddy=None):
        label = QLabel(text)
        if buddy is not None:
            label.setBuddy(buddy)
        form.addRow(label, widget)

    @staticmethod
    def _roi_spin(tooltip):
        spin = QSpinBox()
        spin.setRange(0, 1000000)
        spin.setToolTip(tooltip)
        return spin

    def _connect_signals(self):
        self.model_field.button.clicked.connect(self._browse_model)
        self.input_field.button.clicked.connect(self._browse_input_folder)
        self.overlay_field.button.clicked.connect(self._browse_overlay_folder)
        self.csv_field.button.clicked.connect(self._browse_output_csv)
        self.input_folder_edit.textChanged.connect(self._update_auto_outputs)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        for widget in self._state_widgets():
            signal = widget.textChanged if isinstance(widget, QLineEdit) else None
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                signal = widget.valueChanged
            elif isinstance(widget, QComboBox):
                signal = widget.currentIndexChanged
            if signal is not None:
                signal.connect(lambda *_args: self.state_changed.emit())

    def _state_widgets(self):
        return (
            self.model_edit,
            self.input_folder_edit,
            self.overlay_folder_edit,
            self.output_csv_edit,
            self.confidence_spin,
            self.min_area_spin,
            self.calibration_pixels_spin,
            self.calibration_length_spin,
            self.distance_unit_edit,
            self.roi_x_spin,
            self.roi_y_spin,
            self.roi_width_spin,
            self.roi_height_spin,
            self.edge_policy_combo,
        )

    def serialize_state(self):
        return {
            "model_path": self.model_edit.text(),
            "input_folder": self.input_folder_edit.text(),
            "output_overlay_folder": self.overlay_folder_edit.text(),
            "output_csv": self.output_csv_edit.text(),
            "confidence_threshold": self.confidence_spin.value(),
            "minimum_bubble_area": self.min_area_spin.value(),
            "calibration_pixels": self.calibration_pixels_spin.value(),
            "calibration_length": self.calibration_length_spin.value(),
            "distance_unit": self.distance_unit_edit.text(),
            "roi_x": self.roi_x_spin.value(),
            "roi_y": self.roi_y_spin.value(),
            "roi_width": self.roi_width_spin.value(),
            "roi_height": self.roi_height_spin.value(),
            "edge_bubble_policy": self.edge_policy_combo.currentData(),
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        text_fields = {
            "model_path": self.model_edit,
            "input_folder": self.input_folder_edit,
            "output_overlay_folder": self.overlay_folder_edit,
            "output_csv": self.output_csv_edit,
            "distance_unit": self.distance_unit_edit,
        }
        spin_fields = {
            "confidence_threshold": self.confidence_spin,
            "minimum_bubble_area": self.min_area_spin,
            "calibration_pixels": self.calibration_pixels_spin,
            "calibration_length": self.calibration_length_spin,
            "roi_x": self.roi_x_spin,
            "roi_y": self.roi_y_spin,
            "roi_width": self.roi_width_spin,
            "roi_height": self.roi_height_spin,
        }
        for key, widget in text_fields.items():
            value = state.get(key)
            if value:
                widget.setText(str(value))
        for key, widget in spin_fields.items():
            if key in state:
                widget.setValue(state[key])
        policy = str(state.get("edge_bubble_policy", ""))
        index = self.edge_policy_combo.findData(policy)
        if index >= 0:
            self.edge_policy_combo.setCurrentIndex(index)

    def ensure_defaults(self, model_path="", output_dir=""):
        if model_path and not self.model_edit.text().strip():
            self.model_edit.setText(model_path)
        if output_dir and not self.overlay_folder_edit.text().strip():
            self._last_auto_overlay = osp.join(output_dir, "gas_holdup_overlays")
            self.overlay_folder_edit.setText(self._last_auto_overlay)
        if output_dir and not self.output_csv_edit.text().strip():
            self._last_auto_csv = osp.join(output_dir, "gas_holdup.csv")
            self.output_csv_edit.setText(self._last_auto_csv)

    def _update_auto_outputs(self):
        input_folder = self.input_folder_edit.text().strip()
        if not input_folder:
            return
        normalized = osp.normpath(input_folder)
        parent = osp.dirname(normalized) or "."
        base_name = osp.basename(normalized) or "images"
        next_overlay = osp.join(parent, f"{base_name}_gas_holdup_overlays")
        next_csv = osp.join(parent, f"{base_name}_gas_holdup.csv")
        if not self.overlay_folder_edit.text().strip() or self.overlay_folder_edit.text() == self._last_auto_overlay:
            self.overlay_folder_edit.setText(next_overlay)
        if not self.output_csv_edit.text().strip() or self.output_csv_edit.text() == self._last_auto_csv:
            self.output_csv_edit.setText(next_csv)
        self._last_auto_overlay = next_overlay
        self._last_auto_csv = next_csv

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open segmentation model", "", "PyTorch/ONNX model (*.pt *.onnx);;All files (*.*)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open input image folder")
        if path:
            self.input_folder_edit.setText(path)

    def _browse_overlay_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select overlay output folder")
        if path:
            self.overlay_folder_edit.setText(path)

    def _browse_output_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Gas Holdup CSV", "", "CSV file (*.csv);;All files (*.*)"
        )
        if path:
            self.output_csv_edit.setText(path)

    def _build_config(self):
        model_path = self.model_edit.text().strip()
        input_folder = self.input_folder_edit.text().strip()
        overlay_folder = self.overlay_folder_edit.text().strip()
        output_csv = self.output_csv_edit.text().strip()
        if not model_path or not osp.isfile(model_path):
            raise ValueError("Model file does not exist.")
        if osp.splitext(model_path)[1].lower() not in (".pt", ".onnx"):
            raise ValueError("Model must be a .pt or .onnx file.")
        if not input_folder or not osp.isdir(input_folder):
            raise ValueError("Input folder does not exist.")
        if not overlay_folder:
            raise ValueError("Output Overlay Folder is required.")
        if not output_csv:
            raise ValueError("Output CSV is required.")
        config = GasHoldupConfig(
            model_path=model_path,
            input_folder=input_folder,
            output_overlay_folder=overlay_folder,
            output_csv=output_csv,
            confidence_threshold=self.confidence_spin.value(),
            min_bubble_area=self.min_area_spin.value(),
            calibration_pixels=self.calibration_pixels_spin.value(),
            calibration_length=self.calibration_length_spin.value(),
            distance_unit=self.distance_unit_edit.text().strip(),
            roi=ROI(
                x=self.roi_x_spin.value(),
                y=self.roi_y_spin.value(),
                width=self.roi_width_spin.value(),
                height=self.roi_height_spin.value(),
            ),
            edge_policy=EdgeBubblePolicy.from_value(self.edge_policy_combo.currentData()),
        )
        config.real_units_per_pixel
        config.normalized_distance_unit
        return config

    def start_processing(self):
        if self.is_running():
            return
        try:
            config = self._build_config()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid parameters", str(exc))
            return

        self._set_running(True)
        self.worker = GasHoldupWorker(config)
        self.worker_thread = QThread(self)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._handle_progress)
        self.worker.finished.connect(self._handle_done)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.start()

    def stop_processing(self):
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Stopping...")

    def is_running(self):
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _on_thread_finished(self):
        self.worker_thread = None
        self.processing_finished.emit()

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status_label.setText("Processing..." if running else "Ready")
        if not running:
            self.progress_bar.reset()

    def _handle_progress(self, payload):
        metrics = payload.get("metrics", {})
        frame_id = int(payload.get("frame_id", 0))
        total_frames = int(payload.get("total_frames", 0))
        self.current_frame_value.setText(payload.get("current_frame", "--"))
        self.bubble_count_value.setText(str(metrics.get("bubble_count", 0)))
        self.bubble_area_value.setText(f"{metrics.get('bubble_area', 0)} px")
        self.roi_area_value.setText(f"{metrics.get('roi_area', 0)} px")
        self.area_ratio_value.setText(f"{metrics.get('bubble_area_ratio', 0.0):.2f} %")
        self.gas_holdup_value.setText(f"{metrics.get('gas_holdup', 0.0):.2f} %")
        distance_unit = metrics.get("distance_unit", "unit")
        self.average_diameter_value.setText(
            f"{metrics.get('average_diameter', 0.0):.4f} {distance_unit}"
        )
        self.average_diameter_px_value.setText(
            f"{metrics.get('average_diameter_px', 0.0):.2f} px"
        )
        self.processing_fps_value.setText(f"{metrics.get('fps', 0.0):.2f}")
        self.progress_bar.setMaximum(max(total_frames, 1))
        self.progress_bar.setValue(frame_id)
        self.progress_label.setText(f"{frame_id}/{total_frames}")
        frame = payload.get("preview_frame")
        if frame is not None:
            self._update_preview(frame)

    def _handle_done(self, summary):
        self._set_running(False)
        self.worker = None
        status = "Stopped." if summary.get("stopped") else "Completed."
        self.status_label.setText(status)
        self.progress_label.setText(
            f"{summary.get('processed_frames', 0)}/{summary.get('total_frames', 0)}"
        )
        QMessageBox.information(
            self,
            "Gas Holdup finished",
            f"{status}\n\nOverlay folder:\n{summary.get('overlay_folder', '')}\n\n"
            f"CSV:\n{summary.get('csv_path', '')}",
        )

    def _handle_error(self, error_message):
        self._set_running(False)
        self.worker = None
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Gas Holdup error", error_message)

    def _update_preview(self, frame_bgr):
        max_width = max(self.preview_label.width() - 8, 320)
        max_height = max(self.preview_label.height() - 8, 240)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            height, width, channels = frame_rgb.shape
        image = QImage(
            frame_rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        )
        self.preview_pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(self.preview_pixmap)
