import unittest

import numpy as np

from bubble_analyzer.gas_holdup import ROI
from flow_warning.measurements import FrameMeasurer
from flow_warning.rules import WarningSettings, aggregate_window, fit_baseline, WarningEngine


def rectangle(top=10, left=10, height=4, width=4, shape=(60, 60)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:top + height, left:left + width] = 1
    return mask


class FlowMeasurementsTest(unittest.TestCase):
    def setUp(self):
        self.measurer = FrameMeasurer(ROI(), scale=0.1, min_area=1, max_distance=15)

    def test_raw_count_and_union_do_not_depend_on_track_maturity(self):
        first = rectangle()
        second = rectangle(left=12)
        result = self.measurer.measure([first, second], (60, 60), 1, 0.0)
        self.assertEqual(result.bubble_count, 2)
        self.assertAlmostEqual(result.area_ratio, 24 / 3600 * 100)
        self.assertEqual(result.velocities, [])

    def test_empty_and_invalid_frames_are_distinct(self):
        empty = self.measurer.measure([], (60, 60), 1, 0.0)
        invalid = self.measurer.measure([], (60, 60), 2, 0.1, error="unreadable")
        self.assertEqual(empty.bubble_count, 0)
        self.assertEqual(empty.area_ratio, 0)
        self.assertIsNone(invalid.bubble_count)
        self.assertIsNone(invalid.area_ratio)
        self.assertFalse(invalid.valid)

    def test_partial_roi_masks_count_but_do_not_supply_size_or_motion(self):
        measurer = FrameMeasurer(ROI(12, 12, 30, 30), scale=1, min_area=1)
        result = measurer.measure([rectangle()], (60, 60), 1, 0.0)
        self.assertEqual(result.bubble_count, 1)
        self.assertEqual(result.diameters, [])
        self.assertEqual(result.samples[0]["reason"], "roi_or_image_edge")

    def test_downward_and_sharp_turn_motion_keep_signed_velocity_and_history(self):
        self.measurer.measure([rectangle(top=20)], (60, 60), 1, 0.0)
        upward = self.measurer.measure([rectangle(top=15)], (60, 60), 2, 0.1)
        downward = self.measurer.measure([rectangle(top=20)], (60, 60), 3, 0.2)
        self.assertAlmostEqual(upward.velocities[0], 5.0)
        self.assertAlmostEqual(downward.velocities[0], -5.0)
        self.assertAlmostEqual(downward.samples[0]["turn_angle"], 180)
        self.assertEqual(upward.samples[0]["track_id"], downward.samples[0]["track_id"])
        self.assertEqual(downward.samples[0]["reason"], "")

    def test_missing_time_and_ambiguous_matches_do_not_become_zero_velocity(self):
        self.measurer.measure([rectangle(left=10), rectangle(left=18)], (60, 60), 1, 0.0)
        result = self.measurer.measure([rectangle(left=14)], (60, 60), 2, 0.1)
        self.assertEqual(result.velocities, [])
        self.assertEqual(result.samples[0]["reason"], "ambiguous_match")
        result = self.measurer.measure([rectangle()], (60, 60), 3, None)
        self.assertEqual(result.velocities, [])
        self.assertEqual(result.samples[0]["reason"], "missing_time")

    def test_time_gap_resets_association(self):
        self.measurer.measure([rectangle()], (60, 60), 1, 0.0)
        result = self.measurer.measure([rectangle(top=9)], (60, 60), 4, 0.3)
        self.assertEqual(result.velocities, [])
        self.assertEqual(result.samples[0]["reason"], "frame_gap")


class WarningRulesTest(unittest.TestCase):
    def setUp(self):
        self.settings = WarningSettings(fps=10, window_seconds=0.5, min_samples=3)

    def frames(self, start=1, area=2, diameter=1, velocity=2, count=5):
        measurer = FrameMeasurer(ROI(), scale=0.1, min_area=1)
        frames = []
        for frame_id in range(start, start + count):
            result = measurer.measure([rectangle()], (60, 60), frame_id, (frame_id - 1) / 10)
            result.area_ratio = area
            result.diameters = [diameter] * 5
            result.velocities = [velocity] * 5
            result.complete_count = 5
            frames.append(result)
        return frames

    def baseline(self):
        return fit_baseline(self.frames(count=20), self.settings)

    def test_baseline_cutoff_is_frozen_and_zero_spread_has_noise_floor(self):
        baseline = self.baseline()
        self.assertEqual(baseline["diameter_cutoff"], 1)
        window = aggregate_window(self.frames(diameter=2), self.settings, baseline["diameter_cutoff"])
        self.assertEqual(window["metrics"]["large_fraction"], 1)
        self.assertEqual(baseline["diameter_cutoff"], 1)
        self.assertGreater(baseline["limits"]["area_ratio"]["spread"], 0)

    def test_persistence_recovery_and_event_evidence_times(self):
        engine = WarningEngine(self.baseline(), self.settings)
        first = engine.update(aggregate_window(self.frames(area=12), self.settings, 1))
        second = engine.update(aggregate_window(self.frames(start=6, area=12), self.settings, 1))
        self.assertEqual(first["groups"]["appearance"]["state"], "attention")
        self.assertEqual(second["groups"]["appearance"]["state"], "warning")
        self.assertEqual(engine.events[0]["start_frame"], 1)
        self.assertEqual(engine.events[0]["trigger_frame"], 10)
        recovered_first = engine.update(aggregate_window(self.frames(start=11), self.settings, 1))
        recovered_second = engine.update(aggregate_window(self.frames(start=16), self.settings, 1))
        self.assertEqual(recovered_first["state"], "warning")
        self.assertEqual(recovered_second["state"], "normal")
        self.assertEqual(engine.events[0]["end_reason"], "recovered")

    def test_invalid_data_interrupts_persistence_and_cannot_recover_alarm(self):
        engine = WarningEngine(self.baseline(), self.settings)
        engine.update(aggregate_window(self.frames(area=12), self.settings, 1))
        invalid_frames = self.frames(start=6, area=12)
        for frame in invalid_frames:
            frame.valid = False
        invalid = engine.update(aggregate_window(invalid_frames, self.settings, 1))
        self.assertEqual(invalid["state"], "insufficient")
        result = engine.update(aggregate_window(self.frames(start=11, area=12), self.settings, 1))
        self.assertEqual(result["state"], "attention")
        self.assertEqual(engine.events, [])

    def test_insufficient_samples_and_partial_window_are_not_normal(self):
        frames = self.frames()
        for frame in frames:
            frame.diameters = []
            frame.velocities = []
        result = WarningEngine(self.baseline(), self.settings).update(aggregate_window(frames, self.settings, 1))
        self.assertEqual(result["groups"]["size"]["state"], "insufficient")
        self.assertEqual(result["groups"]["motion"]["state"], "insufficient")
        self.assertEqual(result["state"], "insufficient")
        partial = aggregate_window(self.frames(count=4), self.settings, 1)
        self.assertIsNone(partial["metrics"]["area_ratio"])

    def test_sparse_motion_coverage_is_insufficient(self):
        frames = self.frames()
        for frame in frames:
            frame.velocities = [1]
            frame.complete_count = 20
        window = aggregate_window(frames, self.settings, 1)
        self.assertIsNone(window["metrics"]["rise_median"])

    def test_partial_tail_preserves_active_event_until_sequence_end(self):
        engine = WarningEngine(self.baseline(), self.settings)
        engine.update(aggregate_window(self.frames(area=12), self.settings, 1))
        engine.update(aggregate_window(self.frames(start=6, area=12), self.settings, 1))
        tail = aggregate_window(self.frames(start=11, count=2), self.settings, 1)
        engine.update(tail)
        self.assertIsNone(engine.events[0]["end_reason"])
        engine.finish(tail, "end_of_sequence")
        self.assertEqual(engine.events[0]["end_reason"], "end_of_sequence")
        self.assertEqual(engine.events[0]["end_frame"], 12)

    def test_nonfinite_time_cannot_supply_valid_window(self):
        frames = self.frames()
        for frame in frames:
            frame.timestamp = float("nan")
        window = aggregate_window(frames, self.settings, 1)
        self.assertIsNone(window["metrics"]["area_ratio"])

    def test_baseline_needs_three_full_windows_and_reliable_timing(self):
        with self.assertRaisesRegex(ValueError, "三个"):
            fit_baseline(self.frames(count=10), self.settings)
        with self.assertRaises(ValueError):
            WarningSettings(fps=0).validate()


if __name__ == "__main__":
    unittest.main()
