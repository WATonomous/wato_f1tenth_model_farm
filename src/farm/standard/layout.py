"""Where data lives (Data Standard, section 3).

All data is under DATA_DIR: the environment variable if set, else `src/data/` in this repository.

    >>> from farm.standard.layout import SessionDir, data_dir
    >>> s = SessionDir(data_dir() / "datasets" / "20261006_e7-hallway_r03")
    >>> s.file("rgb", "20261006_e7-hallway_r03_000412").name
    '20261006_e7-hallway_r03_000412.png'
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import ids
from .jsonio import read_json

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/farm/standard/layout.py -> repository root

# Per-sample folders and the extension of the file each holds.
SAMPLE_DIRS: dict[str, str] = {
    "rgb": ".png",
    "depth": ".png",
    "labels": ".txt",
    "masks": ".png",
    "meta": ".json",
}
SESSION_FILES = ("session.json", "tracks.json")


def data_dir(path: str | os.PathLike | None = None) -> Path:
    """DATA_DIR: `path` if given, else the DATA_DIR environment variable, else src/data/."""
    if path is not None:
        return Path(path)
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"])
    return REPO_ROOT / "src" / "data"


def datasets_dir(data: str | os.PathLike | None = None) -> Path:
    return data_dir(data) / "datasets"


def recordings_dir(data: str | os.PathLike | None = None) -> Path:
    return data_dir(data) / "recordings"


def predictions_dir(data: str | os.PathLike | None = None) -> Path:
    return data_dir(data) / "predictions"


@dataclass(frozen=True)
class SessionDir:
    """One dataset session folder: datasets/<session_id>/."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def for_id(cls, session_id: str, data: str | os.PathLike | None = None) -> "SessionDir":
        return cls(datasets_dir(data) / session_id)

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def session_json(self) -> Path:
        return self.root / "session.json"

    @property
    def tracks_json(self) -> Path:
        return self.root / "tracks.json"

    def file(self, kind: str, sample_id: str) -> Path:
        """Path of a sample's file: kind is rgb, depth, labels, masks or meta."""
        return self.root / kind / f"{sample_id}{SAMPLE_DIRS[kind]}"

    def sample_ids(self) -> list[str]:
        """Sample IDs that have a meta/ file (the definition of a sample), sorted by ID."""
        meta = self.root / "meta"
        if not meta.is_dir():
            return []
        return sorted(p.stem for p in meta.glob("*.json") if ids.is_sample_id(p.stem))

    def samples(self) -> list[dict]:
        """Every sample's meta record, ordered by rgb_timestamp_ns (section 2.2: not by index)."""
        metas = [read_json(self.file("meta", sid)) for sid in self.sample_ids()]
        return sorted(metas, key=lambda m: (m.get("rgb_timestamp_ns") or 0, m.get("sample_id", "")))
