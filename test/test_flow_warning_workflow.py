import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

from flow_warning.rules import WarningSettings
from flow_warning.workflow import FlowWarningConfig, run_flow_warning


class MovingPredictor:
    task = "segment"

    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        top = 30 - int(image[0, 0, 0])
        size = 8 if int(image[0, 0, 1]) else 4
        mask[top:top + size, 20:20 + size] = 1
        return None, {"mask_arrays": [mask]}


class FlowWarningWorkflowTest(unittest.TestCase):
    def setup_inputs(self, root):
        baseline = root / "baseline"
        source = root / "source"
        baseline.mkdir()
        source.mkdir()
        for index in range(20):
            image = np.zeros((60, 60, 3), dtype=np.uint8)
            image[0, 0, 0] = index
            cv2.imwrite(str(baseline / f"frame{index}.png"), image)
            image[0, 0, 1] = 1
            cv2.imwrite(str(source / f"frame{index}.png"), image)
        model = root / "model.pt"
        model.touch()
        return FlowWarningConfig(
            model_path=str(model), baseline_path=str(baseline), input_path=str(source),
            output_dir=str(root / "output"), min_area=1,
            settings=WarningSettings(fps=10, window_seconds=0.5, min_samples=3),
        )

    def test_processes_every_frame_and_saves_replay_and_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup_inputs(Path(temp_dir))
            predictor = MovingPredictor()
            updates = []
            summary = run_flow_warning(config, updates.append, predictor_factory=lambda _config: predictor)
            output = Path(summary["output_dir"])
            with (output / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
                frames = list(csv.DictReader(handle))
            with (output / "tracks.csv").open(encoding="utf-8-sig", newline="") as handle:
                tracks = list(csv.DictReader(handle))
            baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
            events = json.loads((output / "events.json").read_text(encoding="utf-8"))
            self.assertEqual(predictor.calls, 40)
            self.assertEqual(len(frames), 40)
            self.assertEqual(len(tracks), 40)
            self.assertEqual(len(list((output / "overlays").glob("*.png"))), 20)
            self.assertEqual(len([update for update in updates if update.get("window")]), 4)
            self.assertGreater(len(events), 0)
            self.assertTrue(all(event["end_reason"] == "end_of_sequence" for event in events))
            self.assertTrue(all(Path(event["evidence_path"]).is_file() for event in events))
            self.assertAlmostEqual(baseline["diameter_cutoff"], 2 * np.sqrt(16 / np.pi) * 0.01)
            self.assertEqual(summary["processed_frames"], 20)
            self.assertFalse(summary["stopped"])

    def test_stop_preserves_completed_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup_inputs(Path(temp_dir))
            stop = threading.Event()

            def stop_after_analysis_frame(payload):
                if payload["phase"] == "analysis":
                    stop.set()

            summary = run_flow_warning(
                config, stop_after_analysis_frame, stop,
                predictor_factory=lambda _config: MovingPredictor(),
            )
            self.assertTrue(summary["stopped"])
            self.assertEqual(summary["processed_frames"], 1)
            self.assertTrue((Path(summary["output_dir"]) / "events.json").exists())
            with (Path(summary["output_dir"]) / "windows.csv").open(encoding="utf-8-sig", newline="") as handle:
                windows = list(csv.DictReader(handle))
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["state"], "insufficient")

    def test_missing_masks_do_not_become_empty_frames(self):
        class MissingMaskPredictor:
            task = "segment"

            def predict(self, image):
                return np.ones((1, 6)), {"mask_boxes": np.ones((1, 4))}

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup_inputs(Path(temp_dir))
            predictor = MovingPredictor()

            def predict_with_broken_analysis(image):
                if int(image[0, 0, 1]):
                    return MissingMaskPredictor().predict(image)
                return predictor.predict(image)

            from types import SimpleNamespace
            combined = SimpleNamespace(task="segment", predict=predict_with_broken_analysis)
            summary = run_flow_warning(config, predictor_factory=lambda _config: combined)
            with (Path(summary["output_dir"]) / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["phase"] == "analysis"]
            self.assertTrue(all(row["reason"] == "missing_segmentation_masks" for row in rows))
            self.assertTrue(all(row["area_ratio"] == "" for row in rows))
            self.assertEqual(summary["events"], [])

    def test_bad_frame_is_missing_data_and_frame_index_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup_inputs(Path(temp_dir))
            (Path(config.input_path) / "frame5.png").write_bytes(b"broken")
            summary = run_flow_warning(config, predictor_factory=lambda _config: MovingPredictor())
            with (Path(summary["output_dir"]) / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["phase"] == "analysis"]
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows[5]["valid"], "False")
            self.assertEqual(rows[5]["bubble_count"], "")
            self.assertEqual(rows[6]["frame"], "7")
            self.assertAlmostEqual(float(rows[6]["time"]), 0.6)

    def test_video_uses_explicit_acquisition_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self.setup_inputs(root)
            video = root / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (60, 60))
            for _ in range(20):
                writer.write(np.zeros((60, 60, 3), dtype=np.uint8))
            writer.release()
            config.input_path = str(video)
            summary = run_flow_warning(config, predictor_factory=lambda _config: MovingPredictor())
            with (Path(summary["output_dir"]) / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["phase"] == "analysis"]
            self.assertAlmostEqual(float(rows[-1]["time"]), 1.9)

    def test_rejects_same_source_and_outputs_inside_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup_inputs(Path(temp_dir))
            config.output_dir = str(Path(config.input_path) / "output")
            with self.assertRaises(ValueError):
                run_flow_warning(config)
            config.output_dir = str(Path(temp_dir) / "output")
            config.baseline_path = config.input_path
            with self.assertRaises(ValueError):
                run_flow_warning(config)


if __name__ == "__main__":
    unittest.main()
