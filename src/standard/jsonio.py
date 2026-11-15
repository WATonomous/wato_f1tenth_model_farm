"""JSON and file writing as the Data Standard requires (section 1).

JSON is UTF-8 with a 2-space indent, no comments, and a final newline. Every write goes to a
temporary file in the same folder and is then renamed over the target, so a reader never sees a
half-written file.

    >>> from farm.standard.jsonio import write_json, read_json
    >>> write_json("session.json", {"schema_version": "1.0"})
    >>> read_json("session.json")["schema_version"]
    '1.0'
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dumps(data: Any) -> str:
    """Serialise exactly as the standard writes JSON: 2-space indent, UTF-8, final newline."""
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_bytes(path: str | os.PathLike, data: bytes, *, overwrite: bool = True) -> Path:
    """Write atomically (temporary file, then rename). With overwrite=False an existing file is an error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(f"{path} exists and overwrite=False")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # mkstemp creates 0600 files; give the file the permissions a plain open() would.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        if not overwrite and path.exists():
            raise FileExistsError(f"{path} exists and overwrite=False")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_text(path: str | os.PathLike, text: str, *, overwrite: bool = True) -> Path:
    return write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def write_json(path: str | os.PathLike, data: Any, *, overwrite: bool = True) -> Path:
    return write_text(path, dumps(data), overwrite=overwrite)


def utc_now() -> str:
    """The current time as the standard writes dates: ISO 8601 in UTC, e.g. 2026-10-08T10:12:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_utc_iso(value: Any) -> bool:
    if not isinstance(value, str) or not UTC_ISO_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
