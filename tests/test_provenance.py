"""Annotation provenance (section 10)."""

from __future__ import annotations

import unittest
from pathlib import Path

from farm.standard import jsonio
from farm.standard import provenance as prov

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
TOOL, MEMBER = "sam3-autolabel@0.1.0", "muhtasim001"


def errors(record):
    return {code for severity, code, _ in prov.check_record(record) if severity == "error"}


class ApplyTests(unittest.TestCase):
    def test_full_life(self):
        r = prov.empty_record()
        self.assertTrue(prov.generator_may_write(r))
        r = prov.apply(r, "generated", TOOL, date="2026-11-15T10:00:00Z")
        self.assertEqual((r["status"], r["generator"], r["updated"]), ("generated", TOOL, "2026-11-15T10:00:00Z"))
        self.assertTrue(prov.generator_may_write(r))
        r = prov.apply(r, "reviewed", MEMBER, date="2026-11-15T11:00:00Z")
        self.assertFalse(prov.generator_may_write(r))
        r = prov.apply(r, "corrected", MEMBER, date="2026-11-15T12:00:00Z", note="moved a box")
        r = prov.apply(r, "corrected", MEMBER, date="2026-11-15T12:30:00Z", note="and another")
        r = prov.apply(r, "excluded", MEMBER, date="2026-11-15T13:00:00Z", exclusion_reason="duplicate")
        self.assertEqual(r["exclusion"], {"reason": "duplicate", "note": None})
        self.assertEqual(prov.status_before_exclusion(r), "corrected")
        with self.assertRaises(prov.ProvenanceError):
            prov.apply(r, "reviewed", MEMBER, date="2026-11-15T14:00:00Z", note="restore")  # wrong target
        with self.assertRaises(prov.ProvenanceError):
            prov.apply(r, "corrected", MEMBER, date="2026-11-15T14:00:00Z")  # restore needs a note
        r = prov.apply(r, "corrected", MEMBER, date="2026-11-15T14:00:00Z", note="not a duplicate after all")
        self.assertIsNone(r["exclusion"])
        self.assertEqual(len(r["history"]), 6)
        self.assertEqual(prov.check_record(r), [])

    def test_hand_made_has_no_generator(self):
        r = prov.apply(prov.empty_record(), "corrected", MEMBER, date="2026-11-15T10:00:00Z")
        self.assertIsNone(r["generator"])
        self.assertEqual(prov.check_record(r), [])

    def test_refusals(self):
        gen = prov.apply(prov.empty_record(), "generated", TOOL, date="2026-11-15T10:00:00Z")
        cases = [
            (prov.empty_record(), dict(status="reviewed", by=MEMBER)),         # none -> reviewed
            (gen, dict(status="generated", by=TOOL)),                          # generated -> generated
            (prov.empty_record(), dict(status="generated", by=MEMBER)),       # a member does not "generate"
            (gen, dict(status="reviewed", by=TOOL)),                           # a tool does not review
            (gen, dict(status="excluded", by=MEMBER)),                         # no reason
            (gen, dict(status="excluded", by=MEMBER, exclusion_reason="other")),  # other needs a note
            (gen, dict(status="reviewed", by=MEMBER, date="2026-11-14T10:00:00Z")),  # earlier than history
            (gen, dict(status="reviewed", by=MEMBER, date="yesterday")),
        ]
        for record, kw in cases:
            with self.subTest(kw=kw), self.assertRaises(prov.ProvenanceError):
                prov.apply(record, kw.pop("status"), kw.pop("by"), **{"date": "2026-11-15T11:00:00Z", **kw})
        excluded_first = prov.apply(prov.empty_record(), "excluded", MEMBER, date="2026-11-15T10:00:00Z",
                                    exclusion_reason="corrupt")
        with self.assertRaises(prov.ProvenanceError):  # no restore to 'none' in the standard
            prov.apply(excluded_first, "generated", TOOL, date="2026-11-15T11:00:00Z", note="x")


class CheckTests(unittest.TestCase):
    def test_examples_are_valid(self):
        for meta_file in sorted(EXAMPLES.glob("*/meta/*.json")):
            meta = jsonio.read_json(meta_file)
            for kind, record in meta["annotations"].items():
                with self.subTest(file=meta_file.name, kind=kind):
                    self.assertEqual(prov.check_record(record), [])

    def test_broken_records(self):
        good = prov.apply(prov.empty_record(), "generated", TOOL, date="2026-11-15T10:00:00Z")
        good = prov.apply(good, "reviewed", MEMBER, date="2026-11-15T11:00:00Z")

        def broken(**changes):
            r = {**good, "history": [dict(e) for e in good["history"]]}
            r.update(changes)
            return r

        self.assertIn("provenance_status", errors(broken(status="corrected")))
        self.assertIn("provenance_updated", errors(broken(updated="2026-11-15T12:00:00Z")))
        self.assertIn("provenance_generator", errors(broken(generator=None)))
        self.assertIn("provenance_exclusion", errors(broken(exclusion={"reason": "other", "note": None})))
        self.assertIn("missing_field", errors({k: v for k, v in good.items() if k != "generator"}))
        bad_order = broken()
        bad_order["history"].reverse()
        self.assertTrue({"provenance_order", "provenance_transition"} <= errors(bad_order))
        self.assertIn("provenance_none", errors({**prov.empty_record(), "updated": "2026-11-15T10:00:00Z"}))
        skip = broken()
        skip["history"] = [skip["history"][1]]  # none -> reviewed
        self.assertIn("provenance_transition", errors(skip))


if __name__ == "__main__":
    unittest.main()
