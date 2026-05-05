#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os.path as osp
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from demo_yolo11 import Args, DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_VIDEO_PATH, main as run_tracker


ROOT_DIR = osp.abspath(osp.dirname(__file__))
UI_STATE_PATH = osp.join(ROOT_DIR, ".bubble_tracker_ui_state.json")
INPUT_FILETYPES = [
    ("Supported inputs", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.webp *.bmp *.png *.tif *.tiff"),
    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
    ("Image files", "*.jpg *.jpeg *.webp *.bmp *.png *.tif *.tiff"),
    ("All files", "*.*"),
]


class BubbleTrackerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bubble Tracker UI")
        self.root.geometry("1280x860")
        self.root.minsize(1120, 760)

        self.message_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = None
        self.preview_photo = None

        self._build_style()
        self._build_variables()
        self._load_persisted_state()
        self._build_layout()
        self._ensure_output_paths()
        self._bind_state_persistence()
        self.root.after(120, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6b7a")

    def _build_variables(self):
        default_input = DEFAULT_VIDEO_PATH if osp.isfile(DEFAULT_VIDEO_PATH) else DEFAULT_IMAGE_DIR
        self.model_path_var = tk.StringVar(value=osp.join(ROOT_DIR, "3900.pt"))
        self.input_path_var = tk.StringVar(value=default_input)
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_video_var = tk.StringVar()
        self.output_csv_var = tk.StringVar()

        self.tracker_type_var = tk.StringVar(value="botsort")
        self.device_var = tk.StringVar(value="auto")
        self.predict_conf_var = tk.DoubleVar(value=0.10)
        self.predict_iou_var = tk.DoubleVar(value=0.45)
        self.mask_match_iou_var = tk.DoubleVar(value=0.10)
        self.input_width_var = tk.IntVar(value=640)
        self.input_height_var = tk.IntVar(value=640)

        self.frame_rate_var = tk.DoubleVar(value=30.0)
        self.pixel_scale_var = tk.DoubleVar(value=1.0)
        self.distance_unit_var = tk.StringVar(value="mm")
        self.speed_gap_var = tk.IntVar(value=1)
        self.min_track_frames_var = tk.IntVar(value=5)
        self.max_turn_angle_var = tk.DoubleVar(value=120.0)
        self.edge_exclusion_margin_var = tk.DoubleVar(value=0.0)
        self.trail_length_var = tk.IntVar(value=24)
        self.preview_stride_var = tk.IntVar(value=5)
        self.show_speed_text_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="Idle")
        self.stats_frame_var = tk.StringVar(value="0")
        self.stats_count_var = tk.StringVar(value="0")
        self.stats_avg_speed_var = tk.StringVar(value="0.00 mm/s")
        self.stats_max_speed_var = tk.StringVar(value="0.00 mm/s")
        self.stats_area_var = tk.StringVar(value="0.00 px^2")
        self.stats_tracks_var = tk.StringVar(value="0")
        self.stats_samples_var = tk.StringVar(value="0")

    def _state_variables(self):
        return {
            "model_path": self.model_path_var,
            "input_path": self.input_path_var,
            "output_dir": self.output_dir_var,
            "output_video": self.output_video_var,
            "output_csv": self.output_csv_var,
            "tracker_type": self.tracker_type_var,
            "device": self.device_var,
            "predict_conf": self.predict_conf_var,
            "predict_iou": self.predict_iou_var,
            "mask_match_iou": self.mask_match_iou_var,
            "input_width": self.input_width_var,
            "input_height": self.input_height_var,
            "frame_rate": self.frame_rate_var,
            "pixel_scale": self.pixel_scale_var,
            "distance_unit": self.distance_unit_var,
            "speed_gap": self.speed_gap_var,
            "min_track_frames": self.min_track_frames_var,
            "max_turn_angle": self.max_turn_angle_var,
            "edge_exclusion_margin": self.edge_exclusion_margin_var,
            "trail_length": self.trail_length_var,
            "preview_stride": self.preview_stride_var,
            "show_speed_text": self.show_speed_text_var,
        }

    def _get_persisted_state(self):
        state = {}
        for key, variable in self._state_variables().items():
            value = variable.get()
            state[key] = value.strip() if isinstance(value, str) else value
        return state

    def _save_persisted_state(self, *_args):
        try:
            with open(UI_STATE_PATH, "w", encoding="utf-8") as file:
                json.dump(self._get_persisted_state(), file, ensure_ascii=False, indent=2)
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

        for key, variable in self._state_variables().items():
            value = state.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            variable.set(value.strip() if isinstance(value, str) else value)

    def _ensure_output_paths(self):
        if not self.output_video_var.get().strip() or not self.output_csv_var.get().strip():
            self._refresh_output_suggestions()

    def _bind_state_persistence(self):
        for variable in self._state_variables().values():
            variable.trace_add("write", self._save_persisted_state)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Bubble Tracker", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Run detection, tracking, motion metrics, and result export for video or image sequences.",
            style="Hint.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        self._build_file_panel(container)
        self._build_action_panel(container)

        center = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        center.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        preview_frame = ttk.LabelFrame(center, text="Preview")
        center.add(preview_frame, weight=3)
        self.preview_label = tk.Label(
            preview_frame,
            text="Preview frames will appear here",
            bg="#f3f5f7",
            fg="#5f6b7a",
            anchor="center",
            justify="center",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        stats_frame = ttk.LabelFrame(center, text="Stats")
        center.add(stats_frame, weight=1)
        self._add_stat_row(stats_frame, 0, "Frame", self.stats_frame_var)
        self._add_stat_row(stats_frame, 1, "Tracked", self.stats_count_var)
        self._add_stat_row(stats_frame, 2, "Avg Speed", self.stats_avg_speed_var)
        self._add_stat_row(stats_frame, 3, "Max Speed", self.stats_max_speed_var)
        self._add_stat_row(stats_frame, 4, "Avg Area", self.stats_area_var)
        self._add_stat_row(stats_frame, 5, "Unique Tracks", self.stats_tracks_var)
        self._add_stat_row(stats_frame, 6, "CSV Samples", self.stats_samples_var)

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=False)

        params_tab = ttk.Frame(notebook, padding=0)
        logs_tab = ttk.Frame(notebook, padding=10)
        notebook.add(params_tab, text="Parameters")
        notebook.add(logs_tab, text="Logs")
        self._build_params_tab(params_tab)
        self._build_logs_tab(logs_tab)

    def _build_file_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Files", padding=10)
        panel.pack(fill=tk.X)
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Model").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(panel, textvariable=self.model_path_var).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(panel, text="Browse...", command=self._browse_model).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(panel, text="Input").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(panel, textvariable=self.input_path_var).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        input_buttons = ttk.Frame(panel)
        input_buttons.grid(row=1, column=2, sticky=tk.E)
        ttk.Button(input_buttons, text="File...", command=self._browse_input_file).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(input_buttons, text="Folder...", command=self._browse_input_dir).pack(side=tk.LEFT)

        ttk.Label(panel, text="Output Dir").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(panel, textvariable=self.output_dir_var).grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(panel, text="Browse...", command=self._browse_output_dir).grid(row=2, column=2, padx=4, pady=4)

        ttk.Label(panel, text="Output Video").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(panel, textvariable=self.output_video_var).grid(row=3, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(panel, text="Save As...", command=self._browse_output_video).grid(row=3, column=2, padx=4, pady=4)

        ttk.Label(panel, text="Output CSV").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Entry(panel, textvariable=self.output_csv_var).grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(panel, text="Save As...", command=self._browse_output_csv).grid(row=4, column=2, padx=4, pady=4)

    def _build_action_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.pack(fill=tk.X, pady=(10, 0))

        button_row = ttk.Frame(panel)
        button_row.pack(fill=tk.X)
        self.start_button = ttk.Button(button_row, text="Start", command=self._start_processing)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(button_row, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(button_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(button_row, textvariable=self.progress_var).pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(panel, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(8, 0))

    def _build_params_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self.root.cget("bg"))
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas, padding=10)

        scroll_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure("params_frame", width=event.width),
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", tags=("params_frame",))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._bind_mousewheel(canvas)

        scroll_frame.columnconfigure(0, weight=1)
        scroll_frame.columnconfigure(1, weight=1)

        detection_frame = ttk.LabelFrame(scroll_frame, text="Detection", padding=10)
        detection_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        motion_frame = ttk.LabelFrame(scroll_frame, text="Motion", padding=10)
        motion_frame.grid(row=0, column=1, sticky="nsew")

        self._add_entry_row(detection_frame, 0, "Tracker", widget=ttk.Combobox(
            detection_frame,
            textvariable=self.tracker_type_var,
            values=("botsort", "bytetrack"),
            state="readonly",
        ))
        self._add_entry_row(detection_frame, 1, "Device", widget=ttk.Combobox(
            detection_frame,
            textvariable=self.device_var,
            values=("auto", "cpu", "cuda:0"),
            state="readonly",
        ))
        self._add_entry_row(detection_frame, 2, "Confidence", variable=self.predict_conf_var)
        self._add_entry_row(detection_frame, 3, "NMS IoU", variable=self.predict_iou_var)
        self._add_entry_row(detection_frame, 4, "Mask Match IoU", variable=self.mask_match_iou_var)
        self._add_entry_row(detection_frame, 5, "Input Width", variable=self.input_width_var)
        self._add_entry_row(detection_frame, 6, "Input Height", variable=self.input_height_var)

        self._add_entry_row(motion_frame, 0, "FPS", variable=self.frame_rate_var)
        self._add_entry_row(motion_frame, 1, "Pixel Scale", variable=self.pixel_scale_var)
        self._add_entry_row(motion_frame, 2, "Distance Unit", variable=self.distance_unit_var)
        self._add_entry_row(motion_frame, 3, "Speed Frame Gap", variable=self.speed_gap_var)
        self._add_entry_row(motion_frame, 4, "Min Track Frames", variable=self.min_track_frames_var)
        self._add_entry_row(motion_frame, 5, "Max Turn Angle", variable=self.max_turn_angle_var)
        self._add_entry_row(motion_frame, 6, "Edge Margin(px)", variable=self.edge_exclusion_margin_var)
        self._add_entry_row(motion_frame, 7, "Trail Length", variable=self.trail_length_var)
        self._add_entry_row(motion_frame, 8, "Preview Stride", variable=self.preview_stride_var)
        ttk.Checkbutton(motion_frame, text="Show speed labels", variable=self.show_speed_text_var).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
        )

    def _build_logs_tab(self, parent):
        self.log_text = tk.Text(parent, height=10, wrap="word", state=tk.DISABLED, bg="#fafbfc")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_stat_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Label(parent, textvariable=variable).grid(row=row, column=1, sticky=tk.E, padx=8, pady=6)
        parent.columnconfigure(1, weight=1)

    def _add_entry_row(self, parent, row, label, variable=None, widget=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
        if widget is None:
            widget = ttk.Entry(parent, textvariable=variable)
        widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=5)
        parent.columnconfigure(1, weight=1)

    def _bind_mousewheel(self, canvas):
        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            canvas.yview_scroll(int(-delta / 120), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Open model",
            filetypes=[("PyTorch/ONNX model", "*.pt *.onnx"), ("All files", "*.*")],
        )
        if file_path:
            self.model_path_var.set(file_path)

    def _browse_input_file(self):
        file_path = filedialog.askopenfilename(title="Open input", filetypes=INPUT_FILETYPES)
        if file_path:
            self.input_path_var.set(file_path)
            self._refresh_output_suggestions()
            self._load_video_fps(file_path)

    def _browse_input_dir(self):
        dir_path = filedialog.askdirectory(title="Open image folder")
        if dir_path:
            self.input_path_var.set(dir_path)
            self._refresh_output_suggestions()

    def _browse_output_dir(self):
        dir_path = filedialog.askdirectory(title="Open output folder")
        if dir_path:
            self.output_dir_var.set(dir_path)
            self._refresh_output_suggestions()

    def _browse_output_video(self):
        file_path = filedialog.asksaveasfilename(
            title="Save output video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if file_path:
            self.output_video_var.set(file_path)

    def _browse_output_csv(self):
        file_path = filedialog.asksaveasfilename(
            title="Save output CSV",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self.output_csv_var.set(file_path)

    def _load_video_fps(self, file_path):
        cap = cv2.VideoCapture(file_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.frame_rate_var.set(round(float(fps), 3))
        finally:
            cap.release()

    def _refresh_output_suggestions(self):
        input_path = self.input_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR
        base_name = "bubble_tracking"
        if input_path:
            if osp.isdir(input_path):
                base_name = osp.basename(osp.normpath(input_path)) or base_name
            else:
                base_name = osp.splitext(osp.basename(input_path))[0] or base_name
        self.output_video_var.set(osp.join(output_dir, f"{base_name}_result.mp4"))
        self.output_csv_var.set(osp.join(output_dir, f"{base_name}_motion.csv"))

    def _build_args(self):
        model_path = self.model_path_var.get().strip()
        input_path = self.input_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR
        output_video = self.output_video_var.get().strip()
        output_csv = self.output_csv_var.get().strip()

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
        args.tracker_type = self.tracker_type_var.get().strip()
        args.device = self.device_var.get().strip() or "auto"
        args.predict_conf = float(self.predict_conf_var.get())
        args.predict_iou = float(self.predict_iou_var.get())
        args.mask_match_iou = float(self.mask_match_iou_var.get())
        args.input_size = (int(self.input_height_var.get()), int(self.input_width_var.get()))

        frame_rate = float(self.frame_rate_var.get())
        if frame_rate <= 0:
            raise ValueError("FPS must be greater than 0.")
        min_track_frames = int(self.min_track_frames_var.get())
        if min_track_frames < 1:
            raise ValueError("Min Track Frames must be at least 1.")
        max_turn_angle = float(self.max_turn_angle_var.get())
        if max_turn_angle <= 0 or max_turn_angle > 180:
            raise ValueError("Max Turn Angle must be in the range (0, 180].")

        args.fps = frame_rate
        args.frame_interval_seconds = 1.0 / frame_rate
        args.pixel_to_real_scale = float(self.pixel_scale_var.get())
        args.distance_unit = self.distance_unit_var.get().strip() or "unit"
        args.speed_frame_gap = int(self.speed_gap_var.get())
        args.min_track_frames = min_track_frames
        args.max_turn_angle_deg = max_turn_angle
        args.edge_exclusion_margin_px = float(self.edge_exclusion_margin_var.get())
        args.trail_length = int(self.trail_length_var.get())
        args.preview_stride = int(self.preview_stride_var.get())
        args.show_speed_text = bool(self.show_speed_text_var.get())
        return args

    def _start_processing(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        try:
            args = self._build_args()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        self.stop_event = threading.Event()
        self._set_running(True)
        self._append_log("Processing started.")
        self.worker_thread = threading.Thread(target=self._run_worker, args=(args,), daemon=True)
        self.worker_thread.start()

    def _run_worker(self, args):
        try:
            summary = run_tracker(args, progress_callback=self._queue_progress, stop_event=self.stop_event)
            self.message_queue.put({"type": "done", "summary": summary})
        except Exception:
            self.message_queue.put({"type": "error", "message": traceback.format_exc()})

    def _queue_progress(self, payload):
        self.message_queue.put({"type": "progress", "payload": payload})

    def _stop_processing(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self._append_log("Stop requested. Waiting for current frame to finish.")

    def _set_running(self, running):
        self.start_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)
        self.status_var.set("Processing..." if running else "Ready")
        if not running:
            self.progress_bar.stop()

    def _poll_queue(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _handle_message(self, message):
        msg_type = message.get("type")
        if msg_type == "progress":
            self._handle_progress(message["payload"])
        elif msg_type == "done":
            self._handle_done(message["summary"])
        elif msg_type == "error":
            self._handle_error(message["message"])

    def _handle_progress(self, payload):
        metrics = payload.get("metrics", {})
        frame_id = payload.get("frame_id", 0)
        total_frames = payload.get("total_frames")
        distance_unit = metrics.get("distance_unit", "unit")

        self.stats_frame_var.set(str(frame_id))
        self.stats_count_var.set(str(metrics.get("tracked_count", 0)))
        self.stats_avg_speed_var.set(f"{metrics.get('avg_speed', 0.0):.2f} {distance_unit}/s")
        self.stats_max_speed_var.set(f"{metrics.get('max_speed', 0.0):.2f} {distance_unit}/s")
        self.stats_area_var.set(f"{metrics.get('avg_area_px', 0.0):.2f} px^2")
        self.stats_tracks_var.set(str(metrics.get("unique_tracks", 0)))
        self.stats_samples_var.set(str(metrics.get("samples_written", 0)))

        if total_frames:
            self.progress_bar.configure(mode="determinate", maximum=total_frames, value=frame_id)
            self.progress_var.set(f"{frame_id}/{total_frames}")
        else:
            self.progress_bar.configure(mode="determinate", maximum=max(frame_id, 1), value=frame_id)
            self.progress_var.set(f"{frame_id} frames")

        frame = payload.get("preview_frame")
        if frame is not None:
            self._update_preview(frame)

    def _handle_done(self, summary):
        self._set_running(False)
        stopped = summary.get("stopped", False)
        status_text = "Stopped." if stopped else "Completed."
        self.status_var.set(status_text)
        self.progress_var.set(f"{summary.get('processed_frames', 0)} frames")
        self._append_log(status_text)
        self._append_log(f"Output video: {summary.get('video_path', '')}")
        self._append_log(f"Motion CSV: {summary.get('motion_csv_path', '')}")
        messagebox.showinfo(
            "Processing finished",
            f"{status_text}\n\nOutput video:\n{summary.get('video_path', '')}\n\nMotion CSV:\n{summary.get('motion_csv_path', '')}",
        )

    def _handle_error(self, error_message):
        self._set_running(False)
        self.status_var.set("Error")
        self._append_log("Error:")
        self._append_log(error_message)
        messagebox.showerror("Error", error_message)

    def _update_preview(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        preview_width = max(self.preview_label.winfo_width() - 16, 320)
        preview_height = max(self.preview_label.winfo_height() - 16, 240)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo, text="")

    def _append_log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_close(self):
        self._save_persisted_state()
        if self.worker_thread is not None and self.worker_thread.is_alive():
            if not messagebox.askyesno("Quit", "Processing is still running. Stop and quit?"):
                return
            self._stop_processing()
        self.root.destroy()


def main():
    root = tk.Tk()
    BubbleTrackerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
