"""Identifiers from the Data Standard (sections 1, 2 and 11).

    >>> from farm.standard import ids
    >>> ids.make_sample_id("20261006_e7-hallway_r03", 412)
    '20261006_e7-hallway_r03_000412'
    >>> ids.split_sample_id("20261006_e7-hallway_r03_000412")
    ('20261006_e7-hallway_r03', 412)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_LOCATION = r"[a-z0-9]+(?:-[a-z0-9]+)*"

# Section 2.1 and 2.2, exactly as written there.
SESSION_ID_RE = re.compile(rf"^(\d{{8}})_({_LOCATION})_r(\d{{2}})$")
SAMPLE_ID_RE = re.compile(rf"^(\d{{8}}_{_LOCATION}_r\d{{2}})_(\d{{6}})$")
LOCATION_RE = re.compile(rf"^{_LOCATION}$")
# Section 11: lowercase, hyphens within fields, underscores between them.
RUN_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:_[a-z0-9]+(?:-[a-z0-9]+)*)*$")
# Section 1: tool IDs are <tool-name>@<version>, member IDs are GitHub usernames.
TOOL_ID_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*@\d+(?:\.\d+)*(?:[-+][0-9A-Za-z.-]+)?$")
MEMBER_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")

FRAME_MAX = 999_999


def is_session_id(value: Any) -> bool:
    return isinstance(value, str) and SESSION_ID_RE.match(value) is not None


def is_sample_id(value: Any) -> bool:
    return isinstance(value, str) and SAMPLE_ID_RE.match(value) is not None


def is_run_id(value: Any) -> bool:
    return isinstance(value, str) and RUN_ID_RE.match(value) is not None


def is_tool_id(value: Any) -> bool:
    return isinstance(value, str) and TOOL_ID_RE.match(value) is not None


def is_member_id(value: Any) -> bool:
    return isinstance(value, str) and MEMBER_ID_RE.match(value) is not None


def is_actor_id(value: Any) -> bool:
    """`by` in a provenance history entry: a member ID or a tool ID."""
    return is_member_id(value) or is_tool_id(value)


def parse_session_id(session_id: str) -> tuple[str, str, int]:
    """(date, location, run) of a session ID; raises ValueError if it is not one."""
    m = SESSION_ID_RE.match(session_id) if isinstance(session_id, str) else None
    if not m:
        raise ValueError(f"not a session ID: {session_id!r}")
    return m.group(1), m.group(2), int(m.group(3))


def has_real_date(session_id: str) -> bool:
    """The regex accepts any 8 digits; this also checks they are a calendar date."""
    date, _, _ = parse_session_id(session_id)
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return False
    return True


def make_session_id(date: str, location: str, run: int) -> str:
    session_id = f"{date}_{location}_r{run:02d}"
    if not 1 <= run <= 99 or not is_session_id(session_id):
        raise ValueError(f"not a valid session ID: {session_id!r}")
    return session_id


def make_sample_id(session_id: str, frame_index: int) -> str:
    if not is_session_id(session_id):
        raise ValueError(f"not a session ID: {session_id!r}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 0 <= frame_index <= FRAME_MAX:
        raise ValueError(f"frame index must be an integer in 0..{FRAME_MAX}: {frame_index!r}")
    return f"{session_id}_{frame_index:06d}"


def split_sample_id(sample_id: str) -> tuple[str, int]:
    """(session_id, frame_index) of a sample ID; raises ValueError if it is not one."""
    m = SAMPLE_ID_RE.match(sample_id) if isinstance(sample_id, str) else None
    if not m:
        raise ValueError(f"not a sample ID: {sample_id!r}")
    return m.group(1), int(m.group(2))


def session_of(sample_id: str) -> str:
    return split_sample_id(sample_id)[0]
