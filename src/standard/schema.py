"""Field-level checks for session.json, meta/<sample_id>.json and tracks.json (sections 6, 7, 8.4).

Each check returns (severity, code, where, message) tuples; farm.standard.validate turns them
into a Report with file paths. The rule for missing values (section 1) is that a key is always
present and holds null when the value is unknown, so a missing key is an error.
"""

from __future__ import annotations

import math
from typing import Any

from . import ids, provenance
from .issues import ERROR, WARNING
from .jsonio import is_utc_iso

SCHEMA_VERSION = "1.0"
MAX_PAIR_DT_NS = 10_000_000  # section 4
ENVIRONMENTS = ("hallway", "tube_track")
ALIGNED_BY = ("realsense_driver", "extraction")
CALIBRATION_SOURCES = ("camera_info", "factory", "none")
SUBTYPES = ("rc_car", "f1tenth", None)
STORAGES = ("mcap", "sqlite3")

SESSION_KEYS = frozenset({"schema_version", "session_id", "environment", "location", "recorded_start_ns",
                          "recorded_end_ns", "source", "camera", "image", "depth", "calibration", "pairing",
                          "extraction"})
SOURCE_KEYS = frozenset({"bag_path", "storage", "rgb_topic", "rgb_source_encoding", "rgb_info_topic",
                         "depth_topic", "depth_info_topic"})
CAMERA_KEYS = frozenset({"model", "serial", "firmware"})
IMAGE_KEYS = frozenset({"width", "height"})
SESSION_DEPTH_KEYS = frozenset({"units", "invalid_value", "source_depth_scale_m", "aligned_to_rgb", "aligned_by"})
CALIBRATION_KEYS = frozenset({"source", "rgb_intrinsics", "distortion_model", "distortion_coeffs",
                              "depth_to_rgb", "rgb_to_base_link"})
INTRINSICS_KEYS = frozenset({"fx", "fy", "cx", "cy"})
EXTRINSICS_KEYS = frozenset({"rotation", "translation_m"})
PAIRING_KEYS = frozenset({"max_dt_ns", "rgb_frames", "paired", "rejected"})
EXTRACTION_KEYS = frozenset({"tool", "date", "stride"})

META_KEYS = frozenset({"schema_version", "sample_id", "session_id", "frame_index", "rgb_timestamp_ns",
                       "depth_timestamp_ns", "pair_dt_ns", "image", "depth", "annotations"})
META_DEPTH_KEYS = frozenset({"units", "invalid_value", "valid_fraction"})
ANNOTATION_TYPES = ("detection", "segmentation")

TRACKS_KEYS = frozenset({"schema_version", "session_id", "tracks", "boxes"})
TRACK_KEYS = frozenset({"subtype", "note"})
BOX_KEYS = frozenset({"line", "track_id", "occluded", "truncated", "unclipped_xyxy_px"})

Finding = tuple[str, str, str, str]  # severity, code, where, message


def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) and math.isfinite(v)


def _keys(doc: Any, keys: frozenset[str], where: str, out: list[Finding]) -> bool:
    """Check an object's keys; False (and an error) if it is not an object at all."""
    if not isinstance(doc, dict):
        out.append((ERROR, "field_type", where, "is not an object"))
        return False
    for key in sorted(keys - doc.keys()):
        out.append((ERROR, "missing_field", where, f"has no '{key}' (unknown values are null, never omitted)"))
    for key in sorted(doc.keys() - keys):
        out.append((WARNING, "unknown_field", where, f"has a field the standard does not define: '{key}'"))
    return True


def _numbers(value: Any, n: int) -> bool:
    return isinstance(value, list) and len(value) == n and all(is_number(v) for v in value)


def _transform(value: Any, where: str, out: list[Finding]) -> None:
    if value is None:
        return
    if _keys(value, EXTRINSICS_KEYS, where, out):
        if not _numbers(value.get("rotation"), 9):
            out.append((ERROR, "calibration", where, "rotation must be 9 numbers (row-major 3x3)"))
        if not _numbers(value.get("translation_m"), 3):
            out.append((ERROR, "calibration", where, "translation_m must be 3 numbers"))


def _image(value: Any, where: str, out: list[Finding]) -> tuple[int, int] | None:
    if not _keys(value, IMAGE_KEYS, where, out):
        return None
    w, h = value.get("width"), value.get("height")
    if not (is_int(w) and is_int(h) and w > 0 and h > 0):
        out.append((ERROR, "image_size", where, f"width and height must be positive integers, got {w!r} x {h!r}"))
        return None
    return w, h


def check_session(doc: Any, expected_id: str) -> list[Finding]:
    """session.json against section 6. expected_id is the session folder's name."""
    out: list[Finding] = []
    if not _keys(doc, SESSION_KEYS, "session.json", out):
        return out
    if doc.get("schema_version") != SCHEMA_VERSION:
        out.append((ERROR, "schema_version", "session.json",
                    f"schema_version is {doc.get('schema_version')!r}; this library reads {SCHEMA_VERSION!r}"))
    sid = doc.get("session_id")
    if sid != expected_id:
        out.append((ERROR, "session_id", "session.json", f"session_id {sid!r} does not match the folder name {expected_id!r}"))
    if doc.get("environment") not in ENVIRONMENTS:
        out.append((ERROR, "environment", "session.json", f"environment {doc.get('environment')!r} is not one of {ENVIRONMENTS}"))
    if ids.is_session_id(expected_id):
        _, location, _ = ids.parse_session_id(expected_id)
        if doc.get("location") != location:
            out.append((ERROR, "location", "session.json",
                        f"location {doc.get('location')!r} is not the session ID's location {location!r}"))
    start, end = doc.get("recorded_start_ns"), doc.get("recorded_end_ns")
    for name, value in (("recorded_start_ns", start), ("recorded_end_ns", end)):
        if value is not None and not is_int(value):
            out.append((ERROR, "timestamp", "session.json", f"{name} must be integer nanoseconds or null"))
    if is_int(start) and is_int(end) and end < start:
        out.append((ERROR, "timestamp", "session.json", "recorded_end_ns is before recorded_start_ns"))

    source = doc.get("source")
    if _keys(source, SOURCE_KEYS, "session.json source", out):
        for key in sorted(SOURCE_KEYS):
            if source.get(key) is not None and not isinstance(source.get(key), str):
                out.append((ERROR, "field_type", "session.json source", f"{key} must be a string or null"))
        if source.get("storage") not in STORAGES + (None,):
            out.append((WARNING, "storage", "session.json source", f"storage {source.get('storage')!r} is not mcap or sqlite3"))
    _keys(doc.get("camera"), CAMERA_KEYS, "session.json camera", out)
    _image(doc.get("image"), "session.json image", out)

    depth = doc.get("depth")
    if _keys(depth, SESSION_DEPTH_KEYS, "session.json depth", out):
        if depth.get("units") != "mm":
            out.append((ERROR, "depth_units", "session.json depth", "units must be 'mm' (section 5.2)"))
        if depth.get("invalid_value") != 0:
            out.append((ERROR, "depth_units", "session.json depth", "invalid_value must be 0 (section 5.2)"))
        scale = depth.get("source_depth_scale_m")
        if scale is not None and not (is_number(scale) and scale > 0):
            out.append((ERROR, "depth_units", "session.json depth", "source_depth_scale_m must be a positive number or null"))
        if depth.get("aligned_to_rgb") is not True:
            out.append((ERROR, "depth_alignment", "session.json depth",
                        "aligned_to_rgb must be true: depth is always aligned to RGB (section 4)"))
        if depth.get("aligned_by") not in ALIGNED_BY:
            out.append((ERROR, "depth_alignment", "session.json depth", f"aligned_by must be one of {ALIGNED_BY}"))

    cal = doc.get("calibration")
    if _keys(cal, CALIBRATION_KEYS, "session.json calibration", out):
        source_kind = cal.get("source")
        if source_kind not in CALIBRATION_SOURCES:
            out.append((ERROR, "calibration", "session.json calibration", f"source must be one of {CALIBRATION_SOURCES}"))
        elif source_kind == "none":
            for key in sorted(CALIBRATION_KEYS - {"source"}):
                if cal.get(key) is not None:
                    out.append((ERROR, "calibration", "session.json calibration",
                                f"source is 'none', so {key} must be null"))
        else:
            intr = cal.get("rgb_intrinsics")
            if intr is None:
                out.append((WARNING, "calibration", "session.json calibration",
                            f"source is {source_kind!r} but rgb_intrinsics is null"))
            elif _keys(intr, INTRINSICS_KEYS, "session.json calibration rgb_intrinsics", out):
                if not all(is_number(intr.get(k)) for k in INTRINSICS_KEYS):
                    out.append((ERROR, "calibration", "session.json calibration rgb_intrinsics", "fx, fy, cx, cy must be numbers"))
            coeffs = cal.get("distortion_coeffs")
            if coeffs is not None and not (isinstance(coeffs, list) and all(is_number(c) for c in coeffs)):
                out.append((ERROR, "calibration", "session.json calibration", "distortion_coeffs must be a list of numbers or null"))
        _transform(cal.get("depth_to_rgb"), "session.json calibration depth_to_rgb", out)
        _transform(cal.get("rgb_to_base_link"), "session.json calibration rgb_to_base_link", out)

    pairing = doc.get("pairing")
    if _keys(pairing, PAIRING_KEYS, "session.json pairing", out):
        max_dt = pairing.get("max_dt_ns")
        if not (is_int(max_dt) and 0 <= max_dt <= MAX_PAIR_DT_NS):
            out.append((ERROR, "pairing", "session.json pairing", f"max_dt_ns must be an integer in 0..{MAX_PAIR_DT_NS}"))
        counts = [pairing.get(k) for k in ("rgb_frames", "paired", "rejected")]
        if not all(is_int(c) and c >= 0 for c in counts):
            out.append((ERROR, "pairing", "session.json pairing", "rgb_frames, paired and rejected must be non-negative integers"))
        elif counts[1] + counts[2] != counts[0]:
            out.append((WARNING, "pairing", "session.json pairing", "paired + rejected does not equal rgb_frames"))

    extraction = doc.get("extraction")
    if _keys(extraction, EXTRACTION_KEYS, "session.json extraction", out):
        if not ids.is_tool_id(extraction.get("tool")):
            out.append((ERROR, "tool_id", "session.json extraction", f"tool {extraction.get('tool')!r} is not <tool-name>@<version>"))
        if not is_utc_iso(extraction.get("date")):
            out.append((ERROR, "date", "session.json extraction", "date must be ISO 8601 UTC"))
        stride = extraction.get("stride")
        if not (is_int(stride) and stride >= 1):
            out.append((ERROR, "field_type", "session.json extraction", "stride must be an integer >= 1"))
    return out


def check_meta(doc: Any, expected_id: str, session_id: str, session_image: tuple[int, int] | None,
               max_dt_ns: int | None) -> tuple[list[Finding], tuple[int, int] | None]:
    """A meta record against sections 7, 10 and 12. Returns the findings and the record's image size."""
    out: list[Finding] = []
    where = f"meta/{expected_id}.json"
    if not _keys(doc, META_KEYS, where, out):
        return out, None
    if doc.get("schema_version") != SCHEMA_VERSION:
        out.append((ERROR, "schema_version", where, f"schema_version is {doc.get('schema_version')!r}, not {SCHEMA_VERSION!r}"))
    if doc.get("sample_id") != expected_id:
        out.append((ERROR, "sample_id", where, f"sample_id {doc.get('sample_id')!r} does not match the file name"))
    if doc.get("session_id") != session_id:
        out.append((ERROR, "session_id", where, f"session_id {doc.get('session_id')!r} is not this session ({session_id!r})"))
    if ids.is_sample_id(expected_id) and doc.get("frame_index") != ids.split_sample_id(expected_id)[1]:
        out.append((ERROR, "frame_index", where, f"frame_index {doc.get('frame_index')!r} does not match the sample ID"))

    rgb_ts, depth_ts, dt = doc.get("rgb_timestamp_ns"), doc.get("depth_timestamp_ns"), doc.get("pair_dt_ns")
    if not (is_int(rgb_ts) and is_int(depth_ts) and is_int(dt)):
        out.append((ERROR, "timestamp", where, "rgb_timestamp_ns, depth_timestamp_ns and pair_dt_ns must be integer nanoseconds"))
    else:
        if dt != depth_ts - rgb_ts:
            out.append((ERROR, "pair_dt", where, f"pair_dt_ns {dt} is not depth_timestamp_ns - rgb_timestamp_ns ({depth_ts - rgb_ts})"))
        limit = max_dt_ns if is_int(max_dt_ns) else MAX_PAIR_DT_NS
        if abs(dt) > min(limit, MAX_PAIR_DT_NS):
            out.append((ERROR, "pair_dt", where, f"|pair_dt_ns| = {abs(dt)} is above the session's max_dt_ns ({limit})"))

    image = _image(doc.get("image"), where + " image", out)
    if image and session_image and image != session_image:
        out.append((ERROR, "image_size", where, f"image {image[0]}x{image[1]} differs from session.json {session_image[0]}x{session_image[1]}"))

    depth = doc.get("depth")
    if _keys(depth, META_DEPTH_KEYS, where + " depth", out):
        if depth.get("units") != "mm" or depth.get("invalid_value") != 0:
            out.append((ERROR, "depth_units", where, "depth units must be 'mm' and invalid_value 0"))
        vf = depth.get("valid_fraction")
        if not (is_number(vf) and 0 <= vf <= 1):
            out.append((ERROR, "valid_fraction", where, f"depth.valid_fraction {vf!r} is not a number in [0, 1]"))

    annotations = doc.get("annotations")
    if _keys(annotations, frozenset(ANNOTATION_TYPES), where + " annotations", out):
        for kind in ANNOTATION_TYPES:
            if kind in annotations:
                for severity, code, message in provenance.check_record(annotations[kind]):
                    out.append((severity, code, f"{where} annotations.{kind}", message))
    return out, image


def check_tracks(doc: Any, session_id: str) -> list[Finding]:
    """tracks.json structure (section 8.4). Per-sample box lists are checked against label files by validate."""
    out: list[Finding] = []
    if not _keys(doc, TRACKS_KEYS, "tracks.json", out):
        return out
    if doc.get("schema_version") != SCHEMA_VERSION:
        out.append((ERROR, "schema_version", "tracks.json", f"schema_version is {doc.get('schema_version')!r}, not {SCHEMA_VERSION!r}"))
    if doc.get("session_id") != session_id:
        out.append((ERROR, "session_id", "tracks.json", f"session_id {doc.get('session_id')!r} is not this session ({session_id!r})"))
    tracks = doc.get("tracks")
    if not isinstance(tracks, dict):
        out.append((ERROR, "field_type", "tracks.json", "tracks is not an object"))
    else:
        for key, track in tracks.items():
            where = f"tracks.json tracks.{key}"
            if not (key.isdigit() and int(key) > 0 and str(int(key)) == key):
                out.append((ERROR, "track_id", where, "track IDs are positive integers"))
            if _keys(track, TRACK_KEYS, where, out):
                if track.get("subtype") not in SUBTYPES:
                    out.append((ERROR, "subtype", where, f"subtype {track.get('subtype')!r} is not rc_car, f1tenth or null"))
                if track.get("note") is not None and not isinstance(track.get("note"), str):
                    out.append((ERROR, "field_type", where, "note must be a string or null"))
    if not isinstance(doc.get("boxes"), dict):
        out.append((ERROR, "field_type", "tracks.json", "boxes is not an object"))
    return out


def check_box_entries(entries: Any, n_lines: int, tracks: dict, sample_id: str,
                      image: tuple[int, int] | None) -> list[Finding]:
    """A sample's tracks.json boxes list against its label file (section 12, annotation files)."""
    out: list[Finding] = []
    where = f"tracks.json boxes.{sample_id}"
    if not isinstance(entries, list):
        out.append((ERROR, "tracks_entry", where, "is not a list"))
        return out
    if len(entries) != n_lines:
        out.append((ERROR, "tracks_lines", where, f"has {len(entries)} entries but the label file has {n_lines} lines"))
    seen: set[int] = set()
    for i, entry in enumerate(entries):
        ewhere = f"{where}[{i}]"
        if not _keys(entry, BOX_KEYS, ewhere, out):
            continue
        if entry.get("line") != i:
            out.append((ERROR, "tracks_lines", ewhere, f"line is {entry.get('line')!r}; entries must be lines 0..n-1 in order"))
        tid = entry.get("track_id")
        if not is_int(tid) or str(tid) not in tracks:
            out.append((ERROR, "track_id", ewhere, f"track_id {tid!r} is not a track in tracks.json"))
        elif tid in seen:
            out.append((ERROR, "track_id", ewhere, f"track_id {tid} appears twice in one sample"))
        else:
            seen.add(tid)
        if not isinstance(entry.get("occluded"), bool):
            out.append((ERROR, "field_type", ewhere, "occluded must be true or false"))
        truncated, unclipped = entry.get("truncated"), entry.get("unclipped_xyxy_px")
        if not isinstance(truncated, bool):
            out.append((ERROR, "field_type", ewhere, "truncated must be true or false"))
        elif truncated:
            if not _numbers(unclipped, 4):
                out.append((ERROR, "unclipped", ewhere, "truncated is true, so unclipped_xyxy_px must be [x1, y1, x2, y2]"))
            elif image and not (unclipped[0] < 0 or unclipped[1] < 0 or unclipped[2] > image[0] or unclipped[3] > image[1]):
                out.append((WARNING, "unclipped", ewhere, "truncated is true but unclipped_xyxy_px lies inside the image"))
        elif unclipped is not None:
            out.append((ERROR, "unclipped", ewhere, "truncated is false, so unclipped_xyxy_px must be null"))
    return out


def new_meta(session_id: str, frame_index: int, rgb_timestamp_ns: int, depth_timestamp_ns: int,
             width: int, height: int, valid_fraction: float) -> dict:
    """A meta record for a new sample, with both annotation types at status 'none'."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": ids.make_sample_id(session_id, frame_index),
        "session_id": session_id,
        "frame_index": frame_index,
        "rgb_timestamp_ns": rgb_timestamp_ns,
        "depth_timestamp_ns": depth_timestamp_ns,
        "pair_dt_ns": depth_timestamp_ns - rgb_timestamp_ns,
        "image": {"width": width, "height": height},
        "depth": {"units": "mm", "invalid_value": 0, "valid_fraction": valid_fraction},
        "annotations": {kind: provenance.empty_record() for kind in ANNOTATION_TYPES},
    }


def new_tracks(session_id: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "session_id": session_id, "tracks": {}, "boxes": {}}
