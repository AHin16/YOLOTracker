from pathlib import Path
import threading

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from bubble_analyzer.gas_holdup import ROI
from .plot import MetricPlot
from .rules import GROUPS, LABELS, STATE_LABELS, WarningSettings
from .workflow import FlowWarningConfig, run_flow_warning
from ui_language import LanguageManager


COLORS = {"normal": "#28754d", "attention": "#946000", "warning": "#b63232", "insufficient": "#666666"}


class FlowWarningWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()

    def run(self):
        try:
            self.finished.emit(run_flow_warning(self.config, self._progress, self.stop_event))
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)

    def _progress(self, payload):
        if payload["phase"] == "baseline_ready" or payload.get("window") or payload.get("frame_id", 0) % 5 == 1:
            self.progress.emit(payload)

    def stop(self):
        self.stop_event.set()


class FlowWarningWidget(QWidget):
    state_changed = Signal()
    processing_finished = Signal()

    def __init__(self, parent=None, language=None):
        super().__init__(parent)
        self.language = language or LanguageManager(parent=self)
        self.worker_thread = None
        self.worker = None
        self.output_path = None
        self.preview_pixmap = None
        self.replay_fps = 30
        self.fields = {}
        self.metric_labels = {}
        self.reason_labels = {}
        self.plots = {}
        self._build_ui()
        self.language.register(self)
        for widget in self.fields.values():
            signal = widget.textChanged if isinstance(widget, QLineEdit) else (
                widget.currentTextChanged if isinstance(widget, QComboBox) else widget.valueChanged
            )
            signal.connect(self.state_changed)

    def _build_ui(self):
        root = QVBoxLayout(self)
        intro = QLabel("选择正常片段建立基线，再检查待测序列中气泡出现量、尺寸与运动的持续变化。")
        intro.setWordWrap(True)
        root.addWidget(intro)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.config_panel = QWidget()
        config_layout = QVBoxLayout(self.config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self._files_group())
        config_layout.addWidget(self._parameters_group())
        left_layout.addWidget(self.config_panel)
        confirmation = QWidget()
        confirmation_layout = QHBoxLayout(confirmation)
        confirmation_layout.setContentsMargins(0, 0, 0, 0)
        self.time_confirm = QCheckBox()
        self.time_confirm.setAccessibleName("已核对两段数据的采集帧率和时间连续性")
        self.time_confirm.setToolTip("帧率用于物理时间换算；抽帧后填有效帧率。文件编号间隔不会自动换算成时间。")
        confirmation_label = QLabel("已核对两段数据的采集帧率和时间连续性")
        confirmation_label.setWordWrap(True)
        confirmation_label.setBuddy(self.time_confirm)
        confirmation_layout.addWidget(self.time_confirm, 0, Qt.AlignTop)
        confirmation_layout.addWidget(confirmation_label, 1)
        left_layout.addWidget(confirmation)
        actions = QHBoxLayout()
        self.start_button = QPushButton("开始预警分析")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        left_layout.addLayout(actions)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.status_label = QLabel("就绪 · 阈值需结合实验数据验证")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        self.open_output_button = QPushButton("打开本次结果目录")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        left_layout.addWidget(self.open_output_button)
        left_layout.addStretch()
        scroll.setWidget(left)
        right = QWidget()
        splitter.addWidget(right)
        right_layout = QVBoxLayout(right)
        self.overall_label = QLabel("等待基线与待测数据")
        self.overall_label.setStyleSheet("font-size: 17px; font-weight: 600; padding: 4px;")
        right_layout.addWidget(self.overall_label)
        cards = QHBoxLayout()
        for group in GROUPS:
            box = QGroupBox(LABELS[group])
            layout = QVBoxLayout(box)
            metric = QLabel("数据不足")
            metric.setWordWrap(True)
            reason = QLabel("等待完整时间窗口")
            reason.setWordWrap(True)
            plot = MetricPlot(language=self.language)
            layout.addWidget(metric)
            layout.addWidget(plot)
            layout.addWidget(reason)
            self.metric_labels[group] = metric
            self.reason_labels[group] = reason
            self.plots[group] = plot
            cards.addWidget(box, 1)
        right_layout.addLayout(cards)
        self.preview_label = QLabel("分析后可拖动滑块，或点击事件回看证据帧")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 200)
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setStyleSheet("background: #20242a; color: #c3cad3;")
        right_layout.addWidget(self.preview_label, 1)
        self.replay_label = QLabel("尚无回放帧")
        right_layout.addWidget(self.replay_label)
        self.replay_slider = QSlider(Qt.Horizontal)
        self.replay_slider.setAccessibleName("证据帧回放")
        self.replay_slider.setRange(0, 0)
        self.replay_slider.valueChanged.connect(self._show_frame)
        right_layout.addWidget(self.replay_slider)
        self.event_table = QTableWidget(0, 4)
        self.event_table.setHorizontalHeaderLabels(["异常方向", "开始 / 触发 (s)", "结束", "触发原因（点击回看）"])
        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.event_table.setMaximumHeight(170)
        self.event_table.itemSelectionChanged.connect(self._select_event)
        right_layout.addWidget(self.event_table)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 1000])

    def _path_row(self, form, key, label, kind):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setMinimumWidth(100)
        edit.setAccessibleName(label)
        self.fields[key] = edit
        layout.addWidget(edit, 1)
        choices = ("文件", "目录") if kind == "source" else ("选择",)
        for choice in choices:
            button = QPushButton(choice)
            self.language.set_property(button, "accessibleName", "Select {field} ({kind})",
                                       field=label, kind=choice)
            button.clicked.connect(lambda checked=False, current=edit, mode=kind, text=choice: self._browse(current, mode, text))
            layout.addWidget(button)
        form.addRow(label, row)

    def _files_group(self):
        group = QGroupBox("数据来源")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._path_row(form, "model_path", "分割模型", "model")
        self._path_row(form, "baseline_path", "正常基线", "source")
        self._path_row(form, "input_path", "待测序列", "source")
        self._path_row(form, "output_dir", "输出目录", "directory")
        return group

    def _number(self, form, key, label, value, minimum, maximum, decimals=None):
        widget = QSpinBox() if decimals is None else QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setAccessibleName(label)
        form.addRow(label, widget)
        self.fields[key] = widget
        return widget

    def _parameters_group(self):
        group = QGroupBox("采集与判定参数")
        form = QFormLayout(group)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.fps_spin = self._number(form, "fps", "实际采集帧率 (fps)", 30, 0.001, 100000, 3)
        self._number(form, "pixel_scale", "每像素实际长度", 0.01, 0.000001, 10000, 6)
        unit = QLineEdit("mm")
        unit.setAccessibleName("长度单位")
        form.addRow("长度单位", unit)
        self.fields["distance_unit"] = unit
        self._number(form, "window_seconds", "时间窗口 (s)", 0.5, 0.001, 120, 3)
        self._number(form, "min_samples", "最少尺寸 / 运动观测数", 10, 1, 100000)
        self._number(form, "trigger_windows", "触发持续窗口数", 2, 1, 100)
        self._number(form, "recovery_windows", "恢复持续窗口数", 2, 1, 100)
        self._number(form, "threshold", "偏离尺度倍数", 3, 0.5, 20, 1)
        self._number(form, "min_area", "最小掩膜面积 (px²)", 10, 1, 100000)
        self._number(form, "max_distance", "相邻帧最大关联距离 (px)", 40, 1, 10000, 1)
        self._number(form, "predict_conf", "分割置信度", 0.1, 0.001, 1, 3)
        device = QComboBox()
        device.addItems(["auto", "cpu", "cuda:0"])
        form.addRow("推理设备", device)
        self.fields["device"] = device
        self._number(form, "input_width", "推理宽度", 640, 32, 4096)
        self._number(form, "input_height", "推理高度", 640, 32, 4096)
        roi_row = QWidget()
        roi_layout = QGridLayout(roi_row)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        for index, (key, label) in enumerate((("roi_x", "X"), ("roi_y", "Y"), ("roi_width", "W"), ("roi_height", "H"))):
            widget = QSpinBox()
            widget.setRange(0, 100000)
            widget.setAccessibleName({"W": "ROI Width", "H": "ROI Height"}.get(label, f"ROI {label}"))
            widget.setToolTip("W / H 为 0 时延伸到图像边界；尺寸与运动只使用 ROI 内完整掩膜。")
            roi_layout.addWidget(QLabel(label), index // 2, index % 2 * 2)
            roi_layout.addWidget(widget, index // 2, index % 2 * 2 + 1)
            self.fields[key] = widget
        form.addRow("监测 ROI", roi_row)
        return group

    def _browse(self, edit, kind, choice):
        if kind == "directory" or choice == "目录":
            path = QFileDialog.getExistingDirectory(self, self.language.text("选择目录"), edit.text())
        else:
            filters = "分割模型 (*.pt *.onnx)" if kind == "model" else "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
            path, _ = QFileDialog.getOpenFileName(self, self.language.text("选择文件"), edit.text(), self.language.text(filters))
        if path:
            edit.setText(path)

    def serialize_state(self):
        return {
            key: widget.text() if isinstance(widget, QLineEdit) else (
                widget.currentData() if isinstance(widget, QComboBox) else widget.value()
            ) for key, widget in self.fields.items()
        }

    def restore_state(self, state):
        if not isinstance(state, dict):
            return
        for key, value in state.items():
            widget = self.fields.get(key)
            if widget is None or value is None:
                continue
            try:
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                else:
                    widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
            except (ValueError, TypeError, OverflowError):
                continue

    def ensure_defaults(self, model_path, output_dir):
        if not self.fields["model_path"].text():
            self.fields["model_path"].setText(model_path)
        if not self.fields["output_dir"].text():
            self.fields["output_dir"].setText(str(Path(output_dir) / "flow_warning"))

    def _build_config(self):
        if not self.time_confirm.isChecked():
            raise ValueError("请先核对两段数据的实际采集帧率和时间连续性，并勾选确认。")
        state = self.serialize_state()
        settings = WarningSettings(**{key: state[key] for key in WarningSettings.__dataclass_fields__})
        config = FlowWarningConfig(
            **{key: state[key].strip() for key in ("model_path", "baseline_path", "input_path", "output_dir", "distance_unit")},
            settings=settings, roi=ROI(*(state[key] for key in ("roi_x", "roi_y", "roi_width", "roi_height"))),
            min_area=state["min_area"], max_distance=state["max_distance"], predict_conf=state["predict_conf"],
            device=state["device"], input_size=(state["input_height"], state["input_width"]),
        )
        config.validate()
        return config

    def start_processing(self):
        if self.is_running():
            return
        try:
            config = self._build_config()
        except Exception as exc:
            QMessageBox.critical(self, self.language.text("请检查预警参数"), self.language.text(exc))
            return
        self.output_path = None
        self.preview_pixmap = None
        self.language.set_text(self.preview_label, "正在建立正常基线…")
        self.replay_slider.setRange(0, 0)
        self.event_table.setRowCount(0)
        self.replay_fps = config.settings.fps
        self.open_output_button.setEnabled(False)
        self.language.set_text(self.overall_label, "正在建立正常基线")
        for group, plot in self.plots.items():
            plot.reset()
            self.language.set_text(self.metric_labels[group], "数据不足")
            self.language.set_text(self.reason_labels[group], "等待基线与完整时间窗口")
        self.config_panel.setEnabled(False)
        self.time_confirm.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker_thread = QThread(self)
        self.worker = FlowWarningWorker(config)
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
        self.worker_thread.finished.connect(self._thread_finished)
        self.worker_thread.start()

    def stop_processing(self):
        if self.worker is not None:
            self.worker.stop()
            self.language.set_text(self.status_label, "正在停止并保存已完成结果…")

    def is_running(self):
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def _thread_finished(self):
        self.worker_thread = None
        self.worker = None
        self.config_panel.setEnabled(True)
        self.time_confirm.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.processing_finished.emit()

    @staticmethod
    def _value(metrics, name, unit="", percent=False):
        value = metrics.get(name)
        return "--" if value is None else f"{value * (100 if percent else 1):.2f}{unit}"

    def _handle_progress(self, payload):
        phase = payload.get("phase")
        frame_id = payload.get("frame_id", 0)
        if phase == "baseline_ready":
            baseline = payload["baseline"]
            for group, names in GROUPS.items():
                self.plots[group].limits = baseline["limits"].get(names[0])
                limit = self.plots[group].limits
                self.language.set_property(self.plots[group], "toolTip",
                    f"{LABELS[names[0]]}正常界限：{limit['lower']:.4g} – {limit['upper']:.4g}"
                    if limit else "该方向的基线有效数据不足"
                )
            cutoff = baseline["diameter_cutoff"]
            self.language.set_property(self.metric_labels["size"], "toolTip",
                f"固定大气泡直径界限：{cutoff:.4g} {self.fields['distance_unit'].text()}"
                if cutoff is not None else "正常基线的完整气泡不足，无法确定大气泡界限"
            )
            self.language.set_text(self.status_label, "基线已固定，正在分析待测数据。" if cutoff is not None else "尺寸基线不足；数量分析继续。")
            return
        total = payload.get("total_frames", 0)
        self.progress_bar.setRange(0, total if total else 0)
        self.progress_bar.setValue(frame_id)
        phase_label = "建立基线" if phase == "baseline" else "分析待测序列"
        self.language.set_text(self.status_label, f"{phase_label} · 第 {frame_id} 帧 / {total or '?'}")
        if payload.get("output_dir"):
            self.output_path = Path(payload["output_dir"])
        if payload.get("overlay_path"):
            self.replay_slider.setRange(1, frame_id)
            if not self.replay_slider.isSliderDown():
                self.replay_slider.setValue(frame_id)
        window = payload.get("window")
        if window:
            state = window["state"]
            self.language.set_text(self.overall_label, f"{STATE_LABELS[state]} · {window['start_time']:.2f}–{window['end_time']:.2f} s")
            self.overall_label.setStyleSheet(f"font-size: 17px; font-weight: 600; color: {COLORS[state]}; padding: 4px;")
            metrics = window["metrics"]
            unit = self.fields["distance_unit"].text()
            descriptions = {
                "appearance": f"面积 {self._value(metrics, 'area_ratio', '%')} · 数量 {self._value(metrics, 'bubble_count')}",
                "size": f"大泡 {self._value(metrics, 'large_fraction', '%', True)} · D90 {self._value(metrics, 'd90', ' ' + unit)}",
                "motion": f"向上 {self._value(metrics, 'rise_median', ' ' + unit + '/s')}\n波动 {self._value(metrics, 'velocity_iqr')} · 下行 {self._value(metrics, 'down_fraction', '%', True)}",
            }
            for group, status in window["groups"].items():
                self.language.set_text(self.metric_labels[group], f"{STATE_LABELS[status['state']]}\n{descriptions[group]}")
                self.metric_labels[group].setStyleSheet(f"color: {COLORS[status['state']]}; font-weight: 600;")
                self.language.set_text(self.reason_labels[group], status["reason"])
                self.plots[group].append(window["end_time"], metrics.get(GROUPS[group][0]))
            self.language.set_property(self.reason_labels["motion"], "toolTip",
                f"运动观测数：{window.get('motion_samples', 0)}；可关联完整样本比例：{window.get('motion_coverage', 0):.1%}"
            )
        if "events" in payload:
            self._set_events(payload["events"])

    def _set_events(self, events):
        self.event_table.blockSignals(True)
        self.event_table.setRowCount(len(events))
        reasons = {"recovered": "已恢复", "data_gap": "数据中断", "stopped": "已停止", "end_of_sequence": "序列结束"}
        for row, event in enumerate(events):
            ending = "持续中" if event["end_time"] is None else f"{event['end_time']:.2f}s · {reasons.get(event['end_reason'], '')}"
            texts = [LABELS[event["group"]], f"{event['start_time']:.2f} / {event['trigger_time']:.2f}", ending, event["reason"]]
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                self.language.set_item_text(item, text)
                item.setData(Qt.UserRole, event["evidence_frame"])
                self.event_table.setItem(row, column, item)
        self.event_table.blockSignals(False)

    def _select_event(self):
        selected = self.event_table.selectedItems()
        if selected:
            self.replay_slider.setValue(int(selected[0].data(Qt.UserRole)))

    def _show_frame(self, frame_id):
        if self.output_path is None or frame_id < 1:
            return
        pixmap = QPixmap(str(self.output_path / "overlays" / f"{frame_id:06d}.png"))
        if not pixmap.isNull():
            self.preview_pixmap = pixmap
            self._fit_preview()
            self.language.set_text(self.replay_label, f"第 {frame_id} 帧 · {(frame_id - 1) / self.replay_fps:.3f} s · 采集时间")

    def _fit_preview(self):
        if self.preview_pixmap is not None:
            self.language.forget_text(self.preview_label)
            self.preview_label.setPixmap(self.preview_pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_preview()

    def _handle_done(self, summary):
        self.output_path = Path(summary["output_dir"])
        self.open_output_button.setEnabled(True)
        self.open_output_button.setToolTip(str(self.output_path))
        self._set_events(summary["events"])
        frame_count = summary["processed_frames"]
        self.replay_slider.setRange(1 if frame_count else 0, frame_count)
        self.replay_slider.setValue(frame_count)
        self.language.set_text(self.status_label, f"{'已停止' if summary['stopped'] else '分析完成'} · {frame_count} 帧 · {len(summary['events'])} 条事件")

    def _handle_error(self, message):
        self.language.set_text(self.overall_label, "分析失败 · 请检查输入与参数")
        self.language.set_text(self.status_label, message)
        QMessageBox.critical(self, self.language.text("流态预警分析失败"), self.language.text(message))

    def _open_output(self):
        if self.output_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_path)))
