import unittest
from types import SimpleNamespace
from unittest.mock import patch

from demo_yolo11 import MotionAnalyzer, process_frame


def make_analyzer():
    return MotionAnalyzer(SimpleNamespace(
        frame_interval_seconds=1.0,
        pixel_to_real_scale=1.0,
        min_track_frames=2,
        distance_unit="mm",
    ))


def track(track_id, x, y, area):
    return {"track_id": track_id, "center": (x, y), "area_px": area}


class MotionTaskStatisticsTest(unittest.TestCase):
    def populated_analyzer(self):
        analyzer = make_analyzer()
        analyzer.update(1, [track(1, 0, 100, 100), track(2, 50, 100, 200)])
        analyzer.update(2, [track(1, 0, 96, 100), track(2, 50, 92, 200)])
        return analyzer

    def test_averages_all_observations_and_retains_task_peak(self):
        analyzer = self.populated_analyzer()

        metrics = analyzer.update(3, [track(1, 0, 94, 600)])

        # Three bubble observations: speeds 4, 8, 2 and areas 100, 200, 600.
        self.assertAlmostEqual(metrics["avg_speed"], 14.0 / 3.0)
        self.assertAlmostEqual(metrics["max_speed"], 8.0)
        self.assertAlmostEqual(metrics["avg_area_px"], 300.0)
        self.assertEqual(metrics["tracked_count"], 1)

    def test_empty_frames_and_pruned_tracks_preserve_task_statistics(self):
        analyzer = self.populated_analyzer()

        for frame_id in (3, 100):
            metrics = analyzer.update(frame_id, [])
            self.assertEqual(metrics["tracked_count"], 0)
            self.assertAlmostEqual(metrics["avg_speed"], 6.0)
            self.assertAlmostEqual(metrics["max_speed"], 8.0)
            self.assertAlmostEqual(metrics["avg_area_px"], 150.0)

    def test_summary_includes_latest_task_statistics(self):
        analyzer = self.populated_analyzer()
        analyzer.update(3, [track(1, 0, 94, 600)])

        summary = analyzer.summary()

        self.assertAlmostEqual(summary["avg_speed"], 14.0 / 3.0)
        self.assertAlmostEqual(summary["max_speed"], 8.0)
        self.assertAlmostEqual(summary["avg_area_px"], 300.0)

    def test_new_task_starts_at_zero(self):
        self.populated_analyzer()
        analyzer = make_analyzer()

        for metrics in (analyzer.summary(), analyzer.update(1, [])):
            for key in ("avg_speed", "max_speed", "avg_area_px"):
                self.assertEqual(metrics[key], 0.0)

    def test_immature_and_invalid_tracks_do_not_add_observations(self):
        analyzer = make_analyzer()
        metrics = analyzer.update(1, [track(1, 0, 100, 900)])
        self.assertEqual(metrics["avg_area_px"], 0.0)
        analyzer.update(2, [track(1, 0, 96, 100)])

        metrics = analyzer.update(3, [track(1, 0, 100, 900)])

        self.assertEqual(metrics["tracked_count"], 0)
        self.assertEqual(metrics["avg_speed"], 4.0)
        self.assertEqual(metrics["avg_area_px"], 100.0)

    @patch("demo_yolo11.render_mask_frame", return_value="preview")
    def test_no_detection_processing_retains_statistics(self, _render):
        analyzer = self.populated_analyzer()
        predictor = SimpleNamespace(predict=lambda frame, timer: (None, {"raw_img": frame}))
        timer = SimpleNamespace(stop=lambda average: 0.01)

        _, _, metrics = process_frame(
            object(), predictor, None, timer, SimpleNamespace(), 2, [], analyzer,
        )

        self.assertEqual(metrics["tracked_count"], 0)
        self.assertEqual(metrics["avg_speed"], 6.0)
        self.assertEqual(metrics["max_speed"], 8.0)
        self.assertEqual(metrics["avg_area_px"], 150.0)


if __name__ == "__main__":
    unittest.main()
