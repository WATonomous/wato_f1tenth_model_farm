"""Whole-session and prediction-run validation (section 12), the docs/examples, and the CLI."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from farm import cli
from farm.standard import images, jsonio, predictions
from farm.standard.validate import find_targets, validate_prediction_run, validate_session

from . import helpers
from .helpers import SESSION_ID, sample

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


class SessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.s = helpers.build_session(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def errors(self):
        return validate_session(self.s.root).codes("error")

    def edit_json(self, path, fn):
        doc = jsonio.read_json(path)
        fn(doc)
        jsonio.write_json(path, doc)

    def test_built_session_is_clean(self):
        report = validate_session(self.s.root)
        self.assertEqual(report.issues, [], "\n".join(map(str, report.issues)))

    def test_missing_rgb(self):
        self.s.file("rgb", sample(1)).unlink()
        self.assertIn("missing_file", self.errors())

    def test_orphan_file(self):
        shutil.copy(self.s.file("rgb", sample(0)), self.s.file("rgb", sample(9)))
        self.assertIn("orphan_file", self.errors())

    def test_bad_filename(self):
        shutil.copy(self.s.file("rgb", sample(0)), self.s.root / "rgb" / "frame0.png")
        self.assertIn("bad_filename", self.errors())

    def test_label_file_rules(self):
        shutil.copy(self.s.file("labels", sample(0)), self.s.file("labels", sample(1)))
        self.assertIn("file_not_expected", self.errors())  # status none, file exists
        self.s.file("labels", sample(1)).unlink()
        self.s.file("labels", sample(0)).unlink()
        self.assertIn("file_missing", self.errors())  # status generated, no file

    def test_excluded_file_may_stay(self):
        jsonio.write_text(self.s.file("labels", sample(2)), "")
        self.edit_json(self.s.tracks_json, lambda t: t["boxes"].__setitem__(sample(2), []))
        self.assertEqual(self.errors(), set())

    def test_pair_dt(self):
        self.edit_json(self.s.file("meta", sample(1)), lambda m: m.__setitem__("pair_dt_ns", 5))
        self.assertIn("pair_dt", self.errors())
        self.edit_json(self.s.file("meta", sample(1)), lambda m: m.update(
            depth_timestamp_ns=m["rgb_timestamp_ns"] + 20_000_000, pair_dt_ns=20_000_000))
        self.assertIn("pair_dt", self.errors())

    def test_image_checks(self):
        images.write_rgb(self.s.file("rgb", sample(1)), np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertIn("image_size", self.errors())
        images.write_mask(self.s.file("depth", sample(1)), np.zeros((helpers.H, helpers.W), dtype=np.uint8))
        self.assertIn("image_format", self.errors())  # 8-bit where depth must be 16-bit

    def test_mask_values(self):
        from PIL import Image
        Image.fromarray(np.full((helpers.H, helpers.W), 7, dtype=np.uint8)).save(self.s.file("masks", sample(0)))
        self.assertIn("mask_values", self.errors())

    def test_tracks(self):
        self.edit_json(self.s.tracks_json, lambda t: t["boxes"][sample(0)].pop())
        self.assertIn("tracks_lines", self.errors())

    def test_truncated_needs_unclipped(self):
        self.edit_json(self.s.tracks_json, lambda t: t["boxes"][sample(0)][1].__setitem__("unclipped_xyxy_px", None))
        self.assertIn("unclipped", self.errors())

    def test_duplicate_track_id(self):
        self.edit_json(self.s.tracks_json, lambda t: t["boxes"][sample(0)][1].__setitem__("track_id", 1))
        self.assertIn("track_id", self.errors())

    def test_missing_tracks_json(self):
        self.s.tracks_json.unlink()
        self.assertIn("missing_file", self.errors())

    def test_session_fields(self):
        self.edit_json(self.s.session_json, lambda d: d.__setitem__("environment", "carpark"))
        self.edit_json(self.s.session_json, lambda d: d["depth"].__setitem__("aligned_to_rgb", False))
        self.edit_json(self.s.session_json, lambda d: d.pop("camera"))
        self.assertTrue({"environment", "depth_alignment", "missing_field"} <= self.errors())

    def test_bad_provenance(self):
        self.edit_json(self.s.file("meta", sample(0)),
                       lambda m: m["annotations"]["detection"].__setitem__("status", "reviewed"))
        self.assertIn("provenance_status", self.errors())

    def test_folder_name(self):
        bad = self.root / "not-a-session"
        shutil.copytree(self.s.root, bad)
        self.assertIn("session_id", validate_session(bad).codes("error"))


class ExampleTests(unittest.TestCase):
    """docs/examples follows the standard; only its not-yet-added images (README "Files to add") are missing."""

    MISSING = {
        "20261006_e7-hallway_r03": {"rgb/20261006_e7-hallway_r03_000412.png", "depth/20261006_e7-hallway_r03_000412.png",
                                    "masks/20261006_e7-hallway_r03_000412.png"},
        "20261010_tube-track_r01": {"rgb/20261010_tube-track_r01_000057.png", "depth/20261010_tube-track_r01_000057.png",
                                    "masks/20261010_tube-track_r01_000057.png", "rgb/20261010_tube-track_r01_000058.png",
                                    "depth/20261010_tube-track_r01_000058.png"},
    }

    def test_example_sessions(self):
        for session_id, missing in self.MISSING.items():
            with self.subTest(session=session_id):
                report = validate_session(EXAMPLES / session_id)
                self.assertEqual(report.codes("error") - {"missing_file", "file_missing"}, set(),
                                 "\n".join(map(str, report.errors)))
                self.assertEqual({i.path for i in report.errors}, missing)

    def test_example_prediction_run(self):
        report = validate_prediction_run(EXAMPLES / "predictions" / "20261012_yolo11n-car_v1")
        self.assertEqual(report.issues, [], "\n".join(map(str, report.issues)))

    def test_find_targets(self):
        kinds = [(k, p.name) for k, p in find_targets(EXAMPLES)]
        self.assertEqual(kinds, [("session", "20261006_e7-hallway_r03"), ("session", "20261010_tube-track_r01"),
                                 ("run", "20261012_yolo11n-car_v1")])


class PredictionTests(unittest.TestCase):
    def test_written_run_is_clean_and_order_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "20261115_yolo11n-car_v2"
            jsonio.write_json(run / "run.json", {
                "schema_version": "1.0", "run_id": run.name, "task": "detection", "model": "yolo11n-car",
                "weights": None, "weights_sha256": None, "tracker": None, "position_method": None,
                "conf_threshold": 0.25, "created": "2026-11-15T12:00:00Z", "sessions": [SESSION_ID]})
            lines = [predictions.detection_line(sample(i), 1_000 - i, [predictions.detection([0.5, 0.5, 0.1, 0.1], conf=0.9)])
                     for i in range(3)]
            predictions.write_detections(run / SESSION_ID / "detections.jsonl", lines)
            images.write_mask(run / SESSION_ID / "masks" / f"{sample(0)}.png", np.ones((4, 4), dtype=np.uint8),
                              prediction=True)
            self.assertEqual(validate_prediction_run(run).issues, [])
            text = (run / SESSION_ID / "detections.jsonl").read_text()
            (run / SESSION_ID / "detections.jsonl").write_text("".join(reversed(text.splitlines(True))))
            self.assertIn("order", validate_prediction_run(run).codes("error"))


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(list(args))
        return code, out.getvalue()

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            s = helpers.build_session(Path(d) / "datasets")
            self.assertEqual(self.run_cli("validate", str(s.root))[0], 0)
            self.assertEqual(self.run_cli("validate", d)[0], 0)  # a DATA_DIR
            s.file("rgb", sample(0)).unlink()
            code, out = self.run_cli("validate", "--quiet", str(s.root))
            self.assertEqual(code, 1)
            self.assertIn("missing_file", out)
            empty = Path(d) / "empty"
            empty.mkdir()
            self.assertEqual(self.run_cli("validate", str(empty))[0], 2)


if __name__ == "__main__":
    unittest.main()
