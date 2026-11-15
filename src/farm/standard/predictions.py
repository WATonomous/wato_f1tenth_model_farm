"""Model outputs under predictions/<run_id>/ (Data Standard, section 11).

    >>> from farm.standard import predictions as pred
    >>> line = pred.detection_line("20261006_e7-hallway_r03_000412", 1791311413733412000, [
    ...     pred.detection([0.5121, 0.6190, 0.2298, 0.1840], conf=0.94, track_id=7, position_m=[0.21, 0.08, 2.47])])
    >>> pred.write_detections("predictions/<run_id>/<session_id>/detections.jsonl", [line])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import ids
from .issues import ERROR, WARNING
from .jsonio import is_utc_iso, write_text
from .schema import SCHEMA_VERSION, Finding, _keys, _numbers, is_int, is_number

TASKS = ("detection", "segmentation")
RUN_KEYS = frozenset({"schema_version", "run_id", "task", "model", "weights", "weights_sha256", "tracker",
                      "position_method", "conf_threshold", "created", "sessions"})
LINE_KEYS = frozenset({"sample_id", "rgb_timestamp_ns", "detections"})
DETECTION_KEYS = frozenset({"track_id", "class_id", "bbox_xywhn", "conf", "position_m"})


def detection(bbox_xywhn: list[float], *, conf: float, track_id: int | None = None,
              position_m: list[float] | None = None) -> dict:
    return {"track_id": track_id, "class_id": 0, "bbox_xywhn": list(bbox_xywhn), "conf": conf,
            "position_m": None if position_m is None else list(position_m)}


def detection_line(sample_id: str, rgb_timestamp_ns: int, detections: list[dict]) -> dict:
    """One detections.jsonl line: every processed sample gets one, also with no detections."""
    return {"sample_id": sample_id, "rgb_timestamp_ns": rgb_timestamp_ns, "detections": detections}


def write_detections(path: str | os.PathLike, lines: list[dict], *, overwrite: bool = True) -> Path:
    """Write detections.jsonl, one compact JSON object per line, ordered by rgb_timestamp_ns."""
    ordered = sorted(lines, key=lambda line: (line["rgb_timestamp_ns"], line["sample_id"]))
    text = "".join(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n" for line in ordered)
    return write_text(path, text, overwrite=overwrite)


def check_run(doc: Any, expected_id: str) -> list[Finding]:
    out: list[Finding] = []
    if not _keys(doc, RUN_KEYS, "run.json", out):
        return out
    if doc.get("schema_version") != SCHEMA_VERSION:
        out.append((ERROR, "schema_version", "run.json", f"schema_version is {doc.get('schema_version')!r}, not {SCHEMA_VERSION!r}"))
    run_id = doc.get("run_id")
    if run_id != expected_id:
        out.append((ERROR, "run_id", "run.json", f"run_id {run_id!r} does not match the folder name {expected_id!r}"))
    if not ids.is_run_id(expected_id):
        out.append((ERROR, "run_id", "run.json", f"{expected_id!r} is not a run ID (lowercase, hyphens within fields, underscores between)"))
    if doc.get("task") not in TASKS:
        out.append((ERROR, "task", "run.json", f"task {doc.get('task')!r} is not one of {TASKS}"))
    conf = doc.get("conf_threshold")
    if conf is not None and not (is_number(conf) and 0 <= conf <= 1):
        out.append((ERROR, "conf", "run.json", "conf_threshold must be in [0, 1] or null"))
    if not is_utc_iso(doc.get("created")):
        out.append((ERROR, "date", "run.json", "created must be ISO 8601 UTC"))
    sessions = doc.get("sessions")
    if not (isinstance(sessions, list) and all(ids.is_session_id(s) for s in sessions)):
        out.append((ERROR, "sessions", "run.json", "sessions must be a list of session IDs"))
    for key in ("model", "weights", "weights_sha256", "tracker", "position_method"):
        if doc.get(key) is not None and not isinstance(doc.get(key), str):
            out.append((ERROR, "field_type", "run.json", f"{key} must be a string or null"))
    return out


def check_detection_line(line: Any, where: str, session_id: str) -> list[Finding]:
    out: list[Finding] = []
    if not _keys(line, LINE_KEYS, where, out):
        return out
    sid = line.get("sample_id")
    if not ids.is_sample_id(sid) or ids.session_of(sid) != session_id:
        out.append((ERROR, "sample_id", where, f"sample_id {sid!r} is not a sample of {session_id}"))
    if not is_int(line.get("rgb_timestamp_ns")):
        out.append((ERROR, "timestamp", where, "rgb_timestamp_ns must be integer nanoseconds"))
    dets = line.get("detections")
    if not isinstance(dets, list):
        out.append((ERROR, "field_type", where, "detections must be a list (empty when nothing was found)"))
        return out
    for i, det in enumerate(dets):
        dwhere = f"{where} detections[{i}]"
        if not _keys(det, DETECTION_KEYS, dwhere, out):
            continue
        if det.get("track_id") is not None and not is_int(det.get("track_id")):
            out.append((ERROR, "track_id", dwhere, "track_id must be an integer or null"))
        if det.get("class_id") != 0:
            out.append((ERROR, "class_id", dwhere, "class_id is always 0"))
        box = det.get("bbox_xywhn")
        if not (_numbers(box, 4) and all(0 <= v <= 1 for v in box)):
            out.append((ERROR, "bbox", dwhere, "bbox_xywhn must be 4 numbers in [0, 1]"))
        if not (is_number(det.get("conf")) and 0 <= det["conf"] <= 1):
            out.append((ERROR, "conf", dwhere, "conf must be a number in [0, 1]"))
        pos = det.get("position_m")
        if pos is not None and not _numbers(pos, 3):
            out.append((ERROR, "position", dwhere, "position_m must be [x, y, z] in metres or null"))
    return out


def check_detections_file(text: str, session_id: str) -> list[Finding]:
    out: list[Finding] = []
    last_ts, seen = None, set()
    for n, raw in enumerate(text.splitlines()):
        where = f"{session_id}/detections.jsonl line {n + 1}"
        if not raw.strip():
            out.append((WARNING, "blank_line", where, "is blank"))
            continue
        try:
            line = json.loads(raw)
        except json.JSONDecodeError as e:
            out.append((ERROR, "json", where, f"is not valid JSON: {e}"))
            continue
        out.extend(check_detection_line(line, where, session_id))
        if isinstance(line, dict):
            ts, sid = line.get("rgb_timestamp_ns"), line.get("sample_id")
            if is_int(ts):
                if last_ts is not None and ts < last_ts:
                    out.append((ERROR, "order", where, "lines must be ordered by rgb_timestamp_ns"))
                last_ts = ts
            if sid in seen:
                out.append((ERROR, "duplicate", where, f"{sid} appears on more than one line"))
            seen.add(sid)
    return out
