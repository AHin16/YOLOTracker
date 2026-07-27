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
