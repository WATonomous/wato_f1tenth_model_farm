"""Validate dataset sessions and prediction runs against the Data Standard (section 12).

    >>> from farm.standard.validate import validate_session
    >>> report = validate_session("data/datasets/20261006_e7-hallway_r03")
    >>> report.ok, len(report.errors), len(report.warnings)

From the command line: `farm validate PATH...` (see src/cli.py).

Errors are section 12 rules (and the formats sections 4-11 define); a session with errors must not
be used for training or evaluation. Warnings are format slips the standard asks for in prose but
section 12 does not list.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import ids, images, labels, predictions, provenance, schema
from .issues import Report
from .jsonio import read_json
from .layout import SAMPLE_DIRS, SESSION_FILES, SessionDir

# What each image kind must be (sections 5.1, 5.2, 9.1): (bit depth, PNG colour type, description)
_PNG_KINDS = {
    "rgb": (8, 2, "8-bit 3-channel RGB"),
    "depth": (16, 0, "16-bit 1-channel greyscale"),
    "masks": (8, 0, "8-bit 1-channel greyscale (mode L, not palette or RGB)"),
}


def _load_json(path: Path, report: Report, rel: str):
    try:
        return read_json(path)
    except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
        report.error("json", rel, f"cannot be read as JSON: {e}")
        return None


def _add(report: Report, findings) -> None:
    for severity, code, where, message in findings:
        report.add(severity, code, where, message)


def _check_png(path: Path, kind: str, size: tuple[int, int] | None, report: Report, rel: str) -> bool:
    bits, color, desc = _PNG_KINDS[kind]
    try:
        header = images.png_header(path)
    except (OSError, ValueError) as e:
        report.error("corrupt_image", rel, f"is not a readable PNG: {e}")
        return False
    if header.bit_depth != bits or header.color_type != color:
        report.error("image_format", rel, f"is {header.kind}; it must be {desc}")
        return False
    if size and (header.width, header.height) != size:
        report.error("image_size", rel, f"is {header.width}x{header.height}, not {size[0]}x{size[1]}")
        return False
    return True


def validate_session(session_dir: str | os.PathLike, *, check_mask_values: bool = True) -> Report:
    """Check one dataset session folder (datasets/<session_id>/) against section 12."""
    s = SessionDir(Path(session_dir))
    report = Report(str(s.root))
    sid = s.id
    if not s.root.is_dir():
        report.error("missing_session", s.root, "is not a folder")
        return report
    if not ids.is_session_id(sid):
        report.error("session_id", s.root, f"folder name {sid!r} is not a session ID (<YYYYMMDD>_<location>_r<NN>)")
    elif not ids.has_real_date(sid):
        report.warning("session_id", s.root, f"the date in {sid!r} is not a calendar date")

    # session.json
    session_image, max_dt = None, None
    if not s.session_json.is_file():
        report.error("missing_file", "session.json", "is missing; every session has one (section 6)")
    else:
        doc = _load_json(s.session_json, report, "session.json")
        if doc is not None:
            _add(report, schema.check_session(doc, sid))
            img = doc.get("image") if isinstance(doc, dict) else None
            if isinstance(img, dict) and schema.is_int(img.get("width")) and schema.is_int(img.get("height")):
                session_image = (img["width"], img["height"])
            pairing = doc.get("pairing") if isinstance(doc, dict) else None
            if isinstance(pairing, dict) and schema.is_int(pairing.get("max_dt_ns")):
                max_dt = pairing["max_dt_ns"]

    # Folder contents: only the standard's files, each named <sample_id><ext>
    files: dict[str, set[str]] = {kind: set() for kind in SAMPLE_DIRS}
    for entry in sorted(s.root.iterdir()):
        name = entry.name
        if entry.is_dir() and name in SAMPLE_DIRS:
            ext = SAMPLE_DIRS[name]
            for f in sorted(entry.iterdir()):
                rel = f"{name}/{f.name}"
                if f.name.startswith("."):
                    report.warning("unexpected_entry", rel, "is a hidden file (a leftover temporary file?)")
                    continue
                stem, suffix = f.name[: -len(ext)], f.name[-len(ext):]
                if not f.is_file() or suffix != ext or not ids.is_sample_id(stem):
                    report.error("bad_filename", rel, f"is not named <sample_id>{ext}")
                elif ids.session_of(stem) != sid:
                    report.error("bad_filename", rel, f"belongs to session {ids.session_of(stem)}, not {sid}")
                else:
                    files[name].add(stem)
        elif not (entry.is_file() and name in SESSION_FILES):
            report.warning("unexpected_entry", name, "is not part of the Data Standard's session layout")

    samples = files["meta"]
    for kind in ("rgb", "depth", "labels", "masks"):
        for stem in sorted(files[kind] - samples):
            report.error("orphan_file", f"{kind}/{stem}{SAMPLE_DIRS[kind]}", "has no meta/ file, so it is not a sample")

    # tracks.json (needed as soon as any label file exists)
    tracks_doc = None
    if s.tracks_json.is_file():
        tracks_doc = _load_json(s.tracks_json, report, "tracks.json")
        if tracks_doc is not None:
            _add(report, schema.check_tracks(tracks_doc, sid))
    elif files["labels"]:
        report.error("missing_file", "tracks.json", "is missing but the session has label files (section 8.4)")
    track_ids = tracks_doc.get("tracks") if isinstance(tracks_doc, dict) and isinstance(tracks_doc.get("tracks"), dict) else {}
    track_boxes = tracks_doc.get("boxes") if isinstance(tracks_doc, dict) and isinstance(tracks_doc.get("boxes"), dict) else {}

    # Each sample
    seen_ts: dict[int, str] = {}
    for stem in sorted(samples):
        rel_meta = f"meta/{stem}.json"
        meta = _load_json(s.file("meta", stem), report, rel_meta)
        if meta is None:
            continue
        findings, image = schema.check_meta(meta, stem, sid, session_image, max_dt)
        _add(report, findings)
        size = image or session_image
        ts = meta.get("rgb_timestamp_ns") if isinstance(meta, dict) else None
        if schema.is_int(ts):
            if ts in seen_ts:
                report.error("duplicate_timestamp", rel_meta, f"has the same rgb_timestamp_ns as {seen_ts[ts]}")
            seen_ts.setdefault(ts, stem)

        for kind in ("rgb", "depth"):
            rel = f"{kind}/{stem}.png"
            if stem not in files[kind]:
                report.error("missing_file", rel, f"is missing: every sample has rgb/, depth/ and meta/ files")
            else:
                _check_png(s.file(kind, stem), kind, size, report, rel)

        annotations = meta.get("annotations") if isinstance(meta, dict) else None
        annotations = annotations if isinstance(annotations, dict) else {}
        det = annotations.get("detection", {}).get("status") if isinstance(annotations.get("detection"), dict) else None
        seg = annotations.get("segmentation", {}).get("status") if isinstance(annotations.get("segmentation"), dict) else None

        # Label file (section 8.3) and its tracks.json entry
        rel_label = f"labels/{stem}.txt"
        has_label = stem in files["labels"]
        _file_rule(report, rel_label, has_label, det, "detection")
        if has_label:
            text = s.file("labels", stem).read_text(encoding="utf-8", errors="replace")
            for p in labels.check_label_text(text):
                where = rel_label if p.line is None else f"{rel_label} line {p.line}"
                report.add(p.severity, p.code, where, p.message)
            if tracks_doc is not None:
                if stem not in track_boxes:
                    report.error("tracks_entry", f"tracks.json boxes.{stem}", "is missing: every sample with a label file has an entry")
                else:
                    _add(report, schema.check_box_entries(track_boxes[stem], labels.count_label_lines(text),
                                                          track_ids, stem, size))

        # Mask file (section 9.1)
        rel_mask = f"masks/{stem}.png"
        has_mask = stem in files["masks"]
        _file_rule(report, rel_mask, has_mask, seg, "segmentation")
        if has_mask and _check_png(s.file("masks", stem), "masks", size, report, rel_mask) and check_mask_values:
            bad = images.bad_mask_values(images.read_mask(s.file("masks", stem)))
            if bad:
                report.error("mask_values", rel_mask, f"holds values {bad}; only 0, 1 and 255 are allowed")

    for stem in sorted(set(track_boxes) - files["labels"]):
        report.warning("tracks_entry", f"tracks.json boxes.{stem}", "has boxes but the sample has no label file")
    return report


def _file_rule(report: Report, rel: str, exists: bool, status, kind: str) -> None:
    """Section 10.3: the file exists for generated/reviewed/corrected, not for none; excluded may keep it."""
    if status not in provenance.STATUSES:
        return  # the bad record was already reported
    if status in provenance.FILE_STATUSES and not exists:
        report.error("file_missing", rel, f"is missing but the {kind} status is {status!r}")
    elif status == "none" and exists:
        report.error("file_not_expected", rel, f"exists but the {kind} status is 'none'")


def validate_prediction_run(run_dir: str | os.PathLike, *, check_mask_values: bool = True) -> Report:
    """Check predictions/<run_id>/ against section 11."""
    root = Path(run_dir)
    report = Report(str(root))
    if not (root / "run.json").is_file():
        report.error("missing_file", "run.json", "is missing (section 11.1)")
        return report
    run = _load_json(root / "run.json", report, "run.json")
    if run is None:
        return report
    _add(report, predictions.check_run(run, root.name))
    listed = set(run.get("sessions") or []) if isinstance(run, dict) and isinstance(run.get("sessions"), list) else set()
    task = run.get("task") if isinstance(run, dict) else None
    for entry in sorted(root.iterdir()):
        if entry.name == "run.json":
            continue
        if not (entry.is_dir() and ids.is_session_id(entry.name)):
            report.warning("unexpected_entry", entry.name, "is not a session folder")
            continue
        if entry.name not in listed:
            report.warning("sessions", entry.name, "has predictions but is not listed in run.json sessions")
        det_file = entry / "detections.jsonl"
        if det_file.is_file():
            _add(report, predictions.check_detections_file(det_file.read_text(encoding="utf-8"), entry.name))
        elif task == "detection":
            report.error("missing_file", f"{entry.name}/detections.jsonl", "is missing for a detection run")
        for sub, allowed in (("masks", images.PREDICTION_MASK_VALUES), ("probs", None)):
            folder = entry / sub
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                rel = f"{entry.name}/{sub}/{f.name}"
                if f.suffix != ".png" or not ids.is_sample_id(f.stem):
                    report.error("bad_filename", rel, "is not named <sample_id>.png")
                    continue
                if _check_png(f, "masks", None, report, rel) and allowed and check_mask_values:
                    bad = images.bad_mask_values(images.read_mask(f), allowed)
                    if bad:
                        report.error("mask_values", rel, f"holds values {bad}; predicted masks hold only 0 and 1")
    return report


def find_targets(path: str | os.PathLike) -> list[tuple[str, Path]]:
    """What a path holds: [('session', dir) | ('run', dir)], for a session, a run, datasets/, predictions/ or DATA_DIR."""
    p = Path(path)
    if (p / "run.json").is_file():
        return [("run", p)]
    if (p / "session.json").is_file() or (p / "meta").is_dir() or (ids.is_session_id(p.name) and p.parent.name != "recordings"):
        return [("session", p)]
    targets: list[tuple[str, Path]] = []
    if p.is_dir():
        for d in sorted(p.iterdir()):
            # recordings/<session_id>/ holds bags, not datasets
            if d.is_dir() and not d.name.startswith(".") and d.name != "recordings":
                targets += find_targets(d)
    return targets


def validate_path(path: str | os.PathLike, *, check_mask_values: bool = True) -> list[Report]:
    out = []
    for kind, target in find_targets(path):
        fn = validate_session if kind == "session" else validate_prediction_run
        out.append(fn(target, check_mask_values=check_mask_values))
    return out
