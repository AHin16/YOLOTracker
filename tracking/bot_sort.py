#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import sys

import numpy as np


BOTSORT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "BoT-SORT"))
if BOTSORT_ROOT not in sys.path:
    sys.path.insert(0, BOTSORT_ROOT)

from tracker.bot_sort import BoTSORT as OfficialBoTSORT


class BOTSORT:
    """Adapter that plugs the official BoT-SORT tracker into this project."""

    def __init__(self, args, frame_rate=30):
        self.args = self._fill_defaults(args)
        self.tracker = OfficialBoTSORT(self.args, frame_rate=frame_rate)

    def _fill_defaults(self, args):
        if not hasattr(args, "track_high_thresh"):
            args.track_high_thresh = getattr(args, "track_thresh", 0.6)
        if not hasattr(args, "track_low_thresh"):
            args.track_low_thresh = 0.1
        if not hasattr(args, "new_track_thresh"):
            args.new_track_thresh = max(args.track_high_thresh, getattr(args, "track_thresh", 0.6))
        if not hasattr(args, "proximity_thresh"):
            args.proximity_thresh = 0.5
        if not hasattr(args, "appearance_thresh"):
            args.appearance_thresh = 0.25
        if not hasattr(args, "with_reid"):
            args.with_reid = False
        if not hasattr(args, "cmc_method"):
            args.cmc_method = "sparseOptFlow"
        if not hasattr(args, "mot20"):
            args.mot20 = False
        if not hasattr(args, "ablation"):
            args.ablation = False
        if not hasattr(args, "name"):
            args.name = "custom"
        if not hasattr(args, "device"):
            args.device = "cpu"
        if not hasattr(args, "fast_reid_config"):
            args.fast_reid_config = osp.join(BOTSORT_ROOT, "fast_reid", "configs", "MOT17", "sbs_S50.yml")
        if not hasattr(args, "fast_reid_weights"):
            args.fast_reid_weights = osp.join(BOTSORT_ROOT, "pretrained", "mot17_sbs_S50.pth")
        return args

    def _to_botsort_detections(self, output_results):
        if output_results is None:
            return np.empty((0, 7), dtype=np.float32)

        if hasattr(output_results, "cpu"):
            output_results = output_results.cpu().numpy()
        else:
            output_results = np.asarray(output_results)

        if output_results.size == 0:
            return np.empty((0, 7), dtype=np.float32)

        if output_results.ndim == 1:
            output_results = np.expand_dims(output_results, axis=0)

        if output_results.shape[1] == 7:
            return output_results.astype(np.float32, copy=False)

        if output_results.shape[1] == 6:
            converted = np.ones((output_results.shape[0], 7), dtype=np.float32)
            converted[:, :4] = output_results[:, :4]
            converted[:, 4] = output_results[:, 4]
            converted[:, 5] = 1.0
            converted[:, 6] = output_results[:, 5]
            return converted

        raise ValueError(f"Unsupported detection shape for BoT-SORT: {output_results.shape}")

    def update(self, output_results, raw_img, img_size=None):
        detections = self._to_botsort_detections(output_results)
        return self.tracker.update(detections, raw_img)
