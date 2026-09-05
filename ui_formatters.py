#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math


def format_processing_fps(value):
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return "Processing FPS: --"

    if not math.isfinite(fps) or fps < 0:
        return "Processing FPS: --"
    return f"Processing FPS: {fps:.2f}"


def format_frame_latency_ms(value):
    try:
        latency_ms = float(value)
    except (TypeError, ValueError):
        return "Frame Latency: --"

    if not math.isfinite(latency_ms) or latency_ms < 0:
        return "Frame Latency: --"
    return f"Frame Latency: {latency_ms:.2f} ms"
