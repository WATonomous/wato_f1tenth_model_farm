"""Annotation provenance records (Data Standard, section 10).

Each sample's meta/ record holds one record per annotation type (detection, segmentation). A
record only changes through allowed transitions, and its history is never edited or truncated.
`apply` makes a new record with one more history entry and refuses anything the standard forbids;
`check_record` finds every problem in a record read from disk.

    >>> from farm.standard import provenance as prov
    >>> rec = prov.empty_record()
    >>> rec = prov.apply(rec, "generated", "sam3-autolabel@0.1.0", date="2026-11-14T12:00:00Z")
    >>> rec = prov.apply(rec, "reviewed", "muhtasim001", date="2026-11-15T09:00:00Z")
    >>> rec["status"], rec["generator"]
    ('reviewed', 'sam3-autolabel@0.1.0')
"""

from __future__ import annotations

import copy
from typing import Any

from . import ids
from .issues import ERROR, WARNING
from .jsonio import is_utc_iso, utc_now

STATUSES = ("none", "generated", "reviewed", "corrected", "excluded")
# A label or mask file must exist in these statuses (section 10.3); for "excluded" it may.
FILE_STATUSES = frozenset({"generated", "reviewed", "corrected"})
# Evaluation uses only these (section 10.3).
EVALUATION_STATUSES = frozenset({"reviewed", "corrected"})
EXCLUSION_REASONS = ("pairing_failed", "motion_blur", "corrupt", "duplicate", "other")
# Section 10.3. From "excluded" the only way out is back to the status it had before exclusion.
TRANSITIONS: dict[str, frozenset[str]] = {
    "none": frozenset({"generated", "corrected", "excluded"}),
    "generated": frozenset({"reviewed", "corrected", "excluded"}),
    "reviewed": frozenset({"corrected", "excluded"}),
    "corrected": frozenset({"corrected", "excluded"}),
}
RECORD_KEYS = frozenset({"status", "generator", "updated", "exclusion", "history"})
ENTRY_KEYS = frozenset({"status", "by", "date", "note"})
MEMBER_STATUSES = frozenset({"reviewed", "corrected"})  # a member, not a tool, sets these


class ProvenanceError(ValueError):
    pass


def empty_record() -> dict:
    """A `none` record: nothing produced yet, no label or mask file."""
    return {"status": "none", "generator": None, "updated": None, "exclusion": None, "history": []}


def status_before_exclusion(record: dict) -> str | None:
    """For an excluded record, the status it had before the exclusion (what a restore returns to)."""
    prev, before = "none", None
    for entry in record.get("history", []):
        status = entry.get("status")
        if status == "excluded" and prev != "excluded":
            before = prev
        prev = status
    return before if prev == "excluded" else None


def generator_may_write(record: dict) -> bool:
    """Section 10.3: a generator re-run overwrites only `generated` annotations (and fills `none`)."""
    return record.get("status") in ("none", "generated")


def apply(record: dict, status: str, by: str, *, date: str | None = None, note: str | None = None,
          exclusion_reason: str | None = None, exclusion_note: str | None = None) -> dict:
    """A copy of `record` after one status change. Raises ProvenanceError for anything not allowed.

    - status "excluded" needs exclusion_reason (and exclusion_note when the reason is "other");
    - leaving "excluded" (a restore) must go back to the status before exclusion, with a note;
    - "generated" is set by a tool ID, "reviewed"/"corrected" by a member ID (GitHub username).
    """
    current = record.get("status")
    if current not in STATUSES:
        raise ProvenanceError(f"record has an unknown status {current!r}")
    if status not in STATUSES or status == "none":
        raise ProvenanceError(f"cannot move an annotation to {status!r}")
    if current == "excluded":
        target = status_before_exclusion(record)
        if target == "none":
            raise ProvenanceError(
                "this annotation was excluded before anything was produced; the standard has no "
                "restore to 'none' (a status 'none' record must have an empty history)")
        if status != target:
            raise ProvenanceError(f"an excluded annotation can only be restored to {target!r}, not {status!r}")
        if not note:
            raise ProvenanceError("restoring an excluded annotation needs a note")
    elif status not in TRANSITIONS[current]:
        raise ProvenanceError(f"{current} -> {status} is not an allowed transition")

    if status == "generated" and not ids.is_tool_id(by):
        raise ProvenanceError(f"'generated' is set by a tool ID (<tool-name>@<version>), not {by!r}")
    if status in MEMBER_STATUSES and not ids.is_member_id(by):
        raise ProvenanceError(f"{status!r} is set by a member (GitHub username), not {by!r}")
    if not ids.is_actor_id(by):
        raise ProvenanceError(f"'by' must be a member ID or a tool ID, not {by!r}")

    exclusion = None
    if status == "excluded":
        if exclusion_reason not in EXCLUSION_REASONS:
            raise ProvenanceError(f"exclusion_reason must be one of {EXCLUSION_REASONS}, not {exclusion_reason!r}")
        if exclusion_reason == "other" and not exclusion_note:
            raise ProvenanceError("exclusion reason 'other' needs an exclusion_note")
        exclusion = {"reason": exclusion_reason, "note": exclusion_note}

    date = date or utc_now()
    if not is_utc_iso(date):
        raise ProvenanceError(f"date must be ISO 8601 UTC like 2026-10-08T10:12:00Z, not {date!r}")
    history = record.get("history", [])
    if history and date < history[-1]["date"]:
        raise ProvenanceError(f"date {date} is before the last history entry ({history[-1]['date']})")

    new = copy.deepcopy(record)
    new["history"].append({"status": status, "by": by, "date": date, "note": note})
    new["status"] = status
    new["updated"] = date
    new["exclusion"] = exclusion
    if not history:
        new["generator"] = by if status == "generated" else None
    return new


def check_record(record: Any) -> list[tuple[str, str, str]]:
    """Every problem in a provenance record, as (severity, code, message)."""
    out: list[tuple[str, str, str]] = []

    def add(severity: str, code: str, message: str) -> None:
        out.append((severity, code, message))

    if not isinstance(record, dict):
        add(ERROR, "provenance_type", "is not an object")
        return out
    for key in sorted(RECORD_KEYS - record.keys()):
        add(ERROR, "missing_field", f"has no '{key}' (unknown values are null, never omitted)")
    for key in sorted(record.keys() - RECORD_KEYS):
        add(WARNING, "unknown_field", f"has an unknown field '{key}'")
    status, history = record.get("status"), record.get("history")
    if status not in STATUSES:
        add(ERROR, "provenance_status", f"status {status!r} is not one of {STATUSES}")
        return out
    if not isinstance(history, list):
        add(ERROR, "provenance_history", "history is not a list")
        return out

    if status == "none":
        if history:
            add(ERROR, "provenance_none", "status is 'none' but history is not empty")
        for key in ("generator", "updated", "exclusion"):
            if record.get(key) is not None:
                add(ERROR, "provenance_none", f"status is 'none' but {key} is not null")
        return out
    if not history:
        add(ERROR, "provenance_history", f"status is {status!r} but history is empty")
        return out

    prev, before_exclusion, last_date = "none", None, None
    for i, entry in enumerate(history):
        where = f"history[{i}]"
        if not isinstance(entry, dict):
            add(ERROR, "provenance_history", f"{where} is not an object")
            return out
        for key in sorted(ENTRY_KEYS - entry.keys()):
            add(ERROR, "missing_field", f"{where} has no '{key}'")
        s, by, date, note = entry.get("status"), entry.get("by"), entry.get("date"), entry.get("note")
        if s not in STATUSES or s == "none":
            add(ERROR, "provenance_history", f"{where} has status {s!r}")
            return out
        if not ids.is_actor_id(by):
            add(ERROR, "provenance_by", f"{where} 'by' {by!r} is neither a member ID nor a tool ID")
        elif s == "generated" and not ids.is_tool_id(by):
            add(ERROR, "provenance_by", f"{where} is 'generated' by {by!r}; generated means produced by a tool")
        elif s in MEMBER_STATUSES and not ids.is_member_id(by):
            add(ERROR, "provenance_by", f"{where} is {s!r} by {by!r}; only a member reviews or corrects")
        if not is_utc_iso(date):
            add(ERROR, "provenance_date", f"{where} date {date!r} is not ISO 8601 UTC")
        elif last_date is not None and date < last_date:
            add(ERROR, "provenance_order", f"{where} is dated before the entry above it (history is oldest first)")
        else:
            last_date = date
        if note is not None and not isinstance(note, str):
            add(ERROR, "provenance_note", f"{where} note is not a string or null")
        if prev == "excluded":
            if s != before_exclusion:
                add(ERROR, "provenance_transition",
                    f"{where}: excluded -> {s} (an exclusion can only be undone back to {before_exclusion!r})")
            elif not note:
                add(ERROR, "provenance_transition", f"{where}: restoring an exclusion needs a note")
        elif s not in TRANSITIONS[prev]:
            add(ERROR, "provenance_transition", f"{where}: {prev} -> {s} is not an allowed transition")
        if s == "excluded" and prev != "excluded":
            before_exclusion = prev
        prev = s

    last = history[-1]
    if status != last.get("status"):
        add(ERROR, "provenance_status", f"status {status!r} is not the last history status {last.get('status')!r}")
    if record.get("updated") != last.get("date"):
        add(ERROR, "provenance_updated", f"updated {record.get('updated')!r} is not the last history date {last.get('date')!r}")
    first = history[0]
    expected_generator = first.get("by") if first.get("status") == "generated" else None
    if record.get("generator") != expected_generator:
        add(ERROR, "provenance_generator",
            f"generator is {record.get('generator')!r} but should be {expected_generator!r} "
            "(the tool of the first history entry if it was generated, else null)")

    exclusion = record.get("exclusion")
    if status == "excluded":
        if not isinstance(exclusion, dict):
            add(ERROR, "provenance_exclusion", "status is 'excluded' but exclusion is not set")
        else:
            reason = exclusion.get("reason")
            if reason not in EXCLUSION_REASONS:
                add(ERROR, "provenance_exclusion", f"exclusion reason {reason!r} is not one of {EXCLUSION_REASONS}")
            if "note" not in exclusion:
                add(ERROR, "missing_field", "exclusion has no 'note'")
            elif reason == "other" and not exclusion.get("note"):
                add(ERROR, "provenance_exclusion", "exclusion reason 'other' needs a note")
    elif exclusion is not None:
        add(ERROR, "provenance_exclusion", f"exclusion is set but status is {status!r}")
    return out
