"""The Data Standard (docs/data-standard.md, version 1.0) as a library.

Every script that reads or writes data goes through this package, so the formats stay exactly
as the standard defines them.

| Module | Section | What |
| --- | --- | --- |
| ids | 1, 2, 11 | session, sample, run, tool and member IDs |
| layout | 3 | DATA_DIR and the paths inside a session |
| jsonio | 1 | JSON and atomic file writes, UTC dates |
| images | 5, 9.1, 11.3 | RGB, depth, mask and probability PNGs |
| labels | 8.3 | YOLO label files and box clipping |
| provenance | 10 | annotation status records and their transitions |
| schema | 6, 7, 8.4 | session.json, meta records, tracks.json |
| predictions | 11 | run.json and detections.jsonl |
| validate | 12 | whole-session and whole-run checks |
"""

from . import ids, images, jsonio, labels, layout, predictions, provenance, schema, validate
from .issues import Issue, Report
from .schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION", "Issue", "Report", "ids", "images", "jsonio", "labels", "layout",
           "predictions", "provenance", "schema", "validate"]
