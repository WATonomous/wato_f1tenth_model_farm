"""IDs, JSON writing, images and label files."""

from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from farm.standard import ids, images, jsonio, labels


class IdTests(unittest.TestCase):
    def test_session_and_sample_ids(self):
        self.assertTrue(ids.is_session_id("20261006_e7-hallway_r03"))
        for bad in ("20261006_e7_hallway_r03", "20261006_E7-hallway_r03", "2026106_e7-hallway_r03",
                    "20261006_e7-hallway_r3", "20261006_-hallway_r03", "20261006_e7-hallway-_r03"):
            self.assertFalse(ids.is_session_id(bad), bad)
        self.assertEqual(ids.make_sample_id("20261006_e7-hallway_r03", 412), "20261006_e7-hallway_r03_000412")
        self.assertEqual(ids.split_sample_id("20261006_e7-hallway_r03_000412"), ("20261006_e7-hallway_r03", 412))
        self.assertFalse(ids.is_sample_id("20261006_e7-hallway_r03_412"))
        with self.assertRaises(ValueError):
            ids.make_sample_id("20261006_e7-hallway_r03", 1_000_000)
        with self.assertRaises(ValueError):
            ids.make_sample_id("20261006_e7-hallway_r03", True)
        self.assertFalse(ids.has_real_date("20261340_e7-hallway_r01"))

    def test_run_tool_member_ids(self):
        self.assertTrue(ids.is_run_id("20261012_yolo11n-car_v1"))
        self.assertFalse(ids.is_run_id("20261012_YOLO_v1"))
        self.assertTrue(ids.is_tool_id("yolo-autolabel@0.2.0"))
        self.assertFalse(ids.is_tool_id("yolo-autolabel"))
        self.assertTrue(ids.is_member_id("muhtasim001"))
        self.assertFalse(ids.is_member_id("yolo-autolabel@0.2.0"))
        self.assertFalse(ids.is_member_id("-bad"))


class JsonTests(unittest.TestCase):
    def test_format_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "x.json"
            jsonio.write_json(p, {"b": None, "a": "é"})
            self.assertEqual(p.read_text(encoding="utf-8"), '{\n  "b": null,\n  "a": "é"\n}\n')
            self.assertEqual(list(p.parent.iterdir()), [p])  # no temporary file left behind
            self.assertNotEqual(stat.S_IMODE(p.stat().st_mode), 0o600)  # not mkstemp's private mode
            with self.assertRaises(FileExistsError):
                jsonio.write_json(p, {}, overwrite=False)

    def test_dates(self):
        self.assertTrue(jsonio.is_utc_iso("2026-10-08T10:12:00Z"))
        self.assertTrue(jsonio.is_utc_iso(jsonio.utc_now()))
        for bad in ("2026-10-08T10:12:00", "2026-10-08 10:12:00Z", "2026-13-08T10:12:00Z", None):
            self.assertFalse(jsonio.is_utc_iso(bad), bad)


class ImageTests(unittest.TestCase):
    def test_round_trips_and_headers(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            images.write_rgb(d / "rgb.png", rgb)
            self.assertEqual(images.png_header(d / "rgb.png"), images.PngHeader(5, 4, 8, 2))
            np.testing.assert_array_equal(images.read_rgb(d / "rgb.png"), rgb)

            depth = np.array([[0, 1, 65535], [1000, 2000, 3000]], dtype=np.uint16)
            images.write_depth(d / "depth.png", depth)
            self.assertEqual(images.png_header(d / "depth.png"), images.PngHeader(3, 2, 16, 0))
            np.testing.assert_array_equal(images.read_depth(d / "depth.png"), depth)
            self.assertAlmostEqual(images.depth_valid_fraction(depth), 5 / 6)

            mask = np.array([[0, 1], [255, 1]], dtype=np.uint8)
            images.write_mask(d / "mask.png", mask)
            self.assertEqual(images.png_header(d / "mask.png"), images.PngHeader(2, 2, 8, 0))
            np.testing.assert_array_equal(images.read_mask(d / "mask.png"), mask)

            with self.assertRaises(ValueError):
                images.write_mask(d / "pred.png", mask, prediction=True)  # 255 never in predictions
            with self.assertRaises(ValueError):
                images.write_mask(d / "bad.png", np.full((2, 2), 7, dtype=np.uint8))
            with self.assertRaises(ValueError):
                images.write_depth(d / "bad.png", depth.astype(np.float32))

            images.write_probs(d / "p.png", np.array([[0.0, 0.5], [1.0, 0.25]]))
            np.testing.assert_array_equal(images.read_mask(d / "p.png"), [[0, 128], [255, 64]])
            (d / "not.png").write_bytes(b"nope")
            with self.assertRaises(ValueError):
                images.png_header(d / "not.png")


class LabelTests(unittest.TestCase):
    def test_standard_example(self):
        # The truncated car in the hallway example (section 8.4): unclipped [-42, 250, 96, 318] at 640x480
        box, truncated = labels.Box.from_xyxy_px(-42, 250, 96, 318, width=640, height=480)
        self.assertTrue(truncated)
        self.assertEqual(labels.format_labels([box]), "0 0.075000 0.591667 0.150000 0.141667\n")

    def test_round_trip_and_empty(self):
        boxes = [labels.Box(0.5125, 0.61875, 0.23125, 0.183333)]
        text = labels.format_labels(boxes)
        self.assertEqual(text, "0 0.512500 0.618750 0.231250 0.183333\n")
        self.assertEqual(labels.parse_labels(text)[0].cx, 0.5125)
        self.assertEqual(labels.format_labels([]), "")
        self.assertEqual(labels.parse_labels(""), [])

    def test_problems(self):
        def codes(text):
            return {(p.severity, p.code) for p in labels.check_label_text(text)}

        self.assertEqual(codes("0 0.500000 0.500000 0.100000 0.100000\n"), set())
        self.assertIn(("error", "label_class"), codes("1 0.500000 0.500000 0.100000 0.100000\n"))
        self.assertIn(("error", "label_fields"), codes("0 0.5 0.5 0.1\n"))
        self.assertIn(("error", "label_range"), codes("0 1.500000 0.500000 0.100000 0.100000\n"))
        self.assertIn(("error", "label_blank_line"), codes("0 0.500000 0.500000 0.100000 0.100000\n\n"))
        self.assertIn(("warning", "label_decimals"), codes("0 0.5 0.500000 0.100000 0.100000\n"))
        self.assertIn(("warning", "label_final_newline"), codes("0 0.500000 0.500000 0.100000 0.100000"))
        self.assertIn(("warning", "label_spacing"), codes("0  0.500000 0.500000 0.100000 0.100000\n"))
        self.assertIn(("warning", "label_not_clipped"), codes("0 0.020000 0.500000 0.100000 0.100000\n"))
        with self.assertRaises(labels.LabelFormatError):
            labels.parse_labels("2 0.5 0.5 0.1 0.1\n")


if __name__ == "__main__":
    unittest.main()
