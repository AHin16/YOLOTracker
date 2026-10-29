from dataclasses import dataclass
import math

import numpy as np


GROUPS = {
    "appearance": ("area_ratio",),
    "size": ("large_fraction", "d90"),
    "motion": ("rise_median", "velocity_iqr", "down_fraction"),
}
LABELS = {
    "appearance": "气泡出现量", "size": "大气泡比例", "motion": "运动状态",
    "area_ratio": "面积占比", "large_fraction": "大气泡比例", "d90": "D90",
    "rise_median": "向上速度中位数", "velocity_iqr": "速度四分位距", "down_fraction": "下行比例",
}
STATE_LABELS = {"normal": "正常", "attention": "关注", "warning": "预警", "insufficient": "数据不足"}


@dataclass(frozen=True)
class WarningSettings:
    fps: float = 30
    window_seconds: float = 0.5
    min_samples: int = 10
    trigger_windows: int = 2
    recovery_windows: int = 2
    threshold: float = 3
    pixel_scale: float = 0.01

    @property
    def frames_per_window(self):
        return max(1, round(self.fps * self.window_seconds))

    def validate(self):
        values = (self.fps, self.window_seconds, self.threshold, self.pixel_scale)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("帧率、窗口、阈值和标定必须为有限正数。")
        if self.fps * self.window_seconds < 2:
            raise ValueError("每个时间窗口至少需要两帧。")
        if min(self.min_samples, self.trigger_windows, self.recovery_windows) < 1:
            raise ValueError("样本数和持续窗口数必须大于零。")


def aggregate_window(frames, settings, diameter_cutoff):
    expected = settings.frames_per_window
    valid = [frame for frame in frames if frame.valid and frame.timestamp is not None and math.isfinite(frame.timestamp)]
    complete_window = len(frames) == expected
    usable = complete_window and len(valid) / expected >= 0.8
    diameters = [value for frame in valid for value in frame.diameters]
    velocities = [value for frame in valid for value in frame.velocities]
    complete_count = sum(frame.complete_count for frame in valid)
    motion_coverage = len(velocities) / complete_count if complete_count else 0
    size_usable = usable and len(diameters) >= settings.min_samples and diameter_cutoff is not None
    motion_usable = usable and len(velocities) >= settings.min_samples and motion_coverage >= 0.5
    metrics = {
        "area_ratio": float(np.median([frame.area_ratio for frame in valid])) if usable else None,
        "bubble_count": float(np.mean([frame.bubble_count for frame in valid])) if usable else None,
        "large_fraction": float(np.mean(np.array(diameters) > diameter_cutoff)) if size_usable else None,
        "d90": float(np.percentile(diameters, 90)) if size_usable else None,
        "rise_median": float(np.median(velocities)) if motion_usable else None,
        "velocity_iqr": float(np.percentile(velocities, 75) - np.percentile(velocities, 25)) if motion_usable else None,
        "down_fraction": float(np.mean(np.array(velocities) < 0)) if motion_usable else None,
    }
    return {
        "start_frame": frames[0].frame_id, "end_frame": frames[-1].frame_id,
        "start_time": (frames[0].frame_id - 1) / settings.fps,
        "end_time": frames[-1].frame_id / settings.fps,
        "valid_fraction": len(valid) / expected, "complete": complete_window,
        "size_samples": len(diameters), "motion_samples": len(velocities),
        "motion_coverage": motion_coverage, "metrics": metrics,
    }


def fit_baseline(frames, settings):
    settings.validate()
    diameters = [value for frame in frames if frame.valid for value in frame.diameters]
    cutoff = float(np.percentile(diameters, 90)) if len(diameters) >= settings.min_samples else None
    windows = [
        aggregate_window(frames[offset:offset + settings.frames_per_window], settings, cutoff)
        for offset in range(0, len(frames), settings.frames_per_window)
    ]
    if sum(window["metrics"]["area_ratio"] is not None for window in windows) < 3:
        raise ValueError("正常基线至少需要三个完整有效时间窗口，请增加连续正常数据。")
    floors = {
        "area_ratio": 1.0, "large_fraction": 0.05, "d90": settings.pixel_scale * 2,
        "rise_median": settings.pixel_scale * settings.fps * 0.5,
        "velocity_iqr": settings.pixel_scale * settings.fps * 0.5, "down_fraction": 0.05,
    }
    limits = {}
    for name, floor in floors.items():
        values = [window["metrics"][name] for window in windows if window["metrics"][name] is not None]
        if len(values) < 3:
            continue
        center = float(np.median(values))
        spread = max(float(np.median(np.abs(np.array(values) - center))) * 1.4826, floor)
        limits[name] = {
            "center": center, "spread": spread,
            "lower": center - settings.threshold * spread,
            "upper": center + settings.threshold * spread,
            "two_sided": name in ("area_ratio", "rise_median"),
        }
    return {"diameter_cutoff": cutoff, "limits": limits, "window_count": len(windows)}


class WarningEngine:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.events = []
        self.states = {
            group: {"count": 0, "recovery": 0, "event": None, "start": None}
            for group in GROUPS
        }

    def _close(self, state, window, reason):
        if state["event"] is not None:
            state["event"].update({
                "end_frame": window["end_frame"], "end_time": window["end_time"],
                "end_reason": reason,
            })
        state.update(count=0, recovery=0, event=None, start=None)

    def update(self, window):
        if not window["complete"]:
            groups = {group: {"state": "insufficient", "reason": "末尾不足一个完整时间窗口"} for group in GROUPS}
            return {**window, "groups": groups, "state": "insufficient"}
        groups = {}
        for group, names in GROUPS.items():
            state = self.states[group]
            available = all(window["metrics"][name] is not None and name in self.baseline["limits"] for name in names)
            if not available:
                self._close(state, window, "data_gap")
                groups[group] = {"state": "insufficient", "reason": "基线或当前窗口的有效样本/覆盖率不足"}
                continue
            reasons = []
            scores = []
            for name in names:
                value = window["metrics"][name]
                limit = self.baseline["limits"][name]
                difference = value - limit["center"]
                score = (abs(difference) if limit["two_sided"] else max(0, difference)) / limit["spread"]
                scores.append(score)
                if score > self.settings.threshold:
                    direction = "高于" if difference > 0 else "低于"
                    bound = limit["upper"] if difference > 0 else limit["lower"]
                    reasons.append(f"{LABELS[name]} {value:.4g} {direction}正常界限 {bound:.4g}")
            if reasons:
                if state["count"] == 0:
                    state["start"] = window
                state["count"] += 1
                state["recovery"] = 0
                if state["count"] >= self.settings.trigger_windows and state["event"] is None:
                    start = state["start"]
                    event = {
                        "id": len(self.events) + 1, "group": group,
                        "start_frame": start["start_frame"], "start_time": start["start_time"],
                        "trigger_frame": window["end_frame"], "trigger_time": window["end_time"],
                        "end_frame": None, "end_time": None, "end_reason": None,
                        "reason": "；".join(reasons), "evidence_frame": window["end_frame"],
                    }
                    state["event"] = event
                    self.events.append(event)
                status = "warning" if state["event"] else "attention"
                reason = "；".join(reasons)
            elif state["event"]:
                state["recovery"] = state["recovery"] + 1 if max(scores) <= self.settings.threshold * 2 / 3 else 0
                status, reason = "warning", "等待指标持续回到恢复范围"
                if state["recovery"] >= self.settings.recovery_windows:
                    self._close(state, window, "recovered")
                    status, reason = "normal", "指标已持续恢复"
            else:
                state.update(count=0, recovery=0, start=None)
                status, reason = "normal", "处于正常基线范围"
            groups[group] = {"state": status, "reason": reason}
        statuses = [group["state"] for group in groups.values()]
        overall = next((status for status in ("warning", "attention", "insufficient") if status in statuses), "normal")
        return {**window, "groups": groups, "state": overall}

    def finish(self, window, reason):
        for state in self.states.values():
            self._close(state, window, reason)
