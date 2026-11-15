"""Detection label files, labels/<sample_id>.txt (Data Standard, section 8.3).

One line per car, `<class_id> <cx> <cy> <w> <h>`: the clipped box, normalised by image width and
height, six decimals, single spaces, a newline at the end. An empty file means no cars. Line order
matters: tracks.json refers to boxes by zero-based line index.

    >>> from farm.standard.labels import Box, format_labels, parse_labels
    >>> box, truncated = Box.from_xyxy_px(-42, 250, 96, 318, width=640, height=480)
    >>> truncated
    True
    >>> format_labels([box])
    '0 0.075000 0.591667 0.150000 0.141667\\n'
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .issues import ERROR, WARNING
from .jsonio import write_text

CAR = 0
CLASS_IDS = frozenset({CAR})  # section 8.1: one class
_SIX_DECIMALS = re.compile(r"^-?\d+\.\d{6}$")
_EDGE_TOLERANCE = 1e-6  # six-decimal rounding can put an edge just past 0 or 1


@dataclass(frozen=True)
class Box:
    """A clipped box in normalised YOLO form (centre, width, height in [0, 1])."""

    cx: float
    cy: float
    w: float
    h: float
    class_id: int = CAR

    def xyxy_px(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            (self.cx - self.w / 2) * width,
            (self.cy - self.h / 2) * height,
            (self.cx + self.w / 2) * width,
            (self.cy + self.h / 2) * height,
        )

    @classmethod
    def from_xyxy_px(cls, x1: float, y1: float, x2: float, y2: float, *, width: int, height: int,
                     class_id: int = CAR) -> tuple["Box", bool]:
        """Clip a (possibly amodal, out-of-image) pixel box to the image.

        Returns (box, truncated). truncated is True when the box crossed the image boundary;
        tracks.json then records the unclipped box as unclipped_xyxy_px (section 8.2).
        """
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"box needs x2 > x1 and y2 > y1: {(x1, y1, x2, y2)}")
        truncated = x1 < 0 or y1 < 0 or x2 > width or y2 > height
        cx1, cy1 = min(max(x1, 0.0), width), min(max(y1, 0.0), height)
        cx2, cy2 = min(max(x2, 0.0), width), min(max(y2, 0.0), height)
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(f"box lies entirely outside the {width}x{height} image: {(x1, y1, x2, y2)}")
        box = cls((cx1 + cx2) / 2 / width, (cy1 + cy2) / 2 / height, (cx2 - cx1) / width,
                  (cy2 - cy1) / height, class_id)
        return box, truncated


@dataclass(frozen=True)
class LabelProblem:
    severity: str
    code: str
    line: int | None  # zero-based line index, as tracks.json counts lines
    message: str


def check_label_text(text: str) -> list[LabelProblem]:
    """Every problem in a label file's text. Errors break section 8.3 or 12; warnings are format slips."""
    problems: list[LabelProblem] = []

    def add(severity: str, code: str, line: int | None, message: str) -> None:
        problems.append(LabelProblem(severity, code, line, message))

    if text == "":
        return problems
    if "\r" in text:
        add(WARNING, "label_crlf", None, "uses CR/LF line endings; the standard uses \\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        add(WARNING, "label_final_newline", None, "has no newline at the end of the file")
    lines = text[:-1].split("\n") if text.endswith("\n") else text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "":
            add(ERROR, "label_blank_line", i, "is blank; line indices must match tracks.json, so blank lines are not allowed")
            continue
        fields = line.split(" ")
        if len(fields) != 5:
            if len(line.split()) == 5:
                add(WARNING, "label_spacing", i, "fields are not separated by single spaces")
                fields = line.split()
            else:
                add(ERROR, "label_fields", i, f"has {len(line.split())} fields, not 5 (<class_id> <cx> <cy> <w> <h>)")
                continue
        if fields[0] not in {str(c) for c in CLASS_IDS}:
            add(ERROR, "label_class", i, f"class_id is {fields[0]!r}; the only class is 0 (car)")
        values = []
        for name, token in zip(("cx", "cy", "w", "h"), fields[1:]):
            try:
                value = float(token)
            except ValueError:
                add(ERROR, "label_number", i, f"{name} is not a number: {token!r}")
                break
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                add(ERROR, "label_range", i, f"{name} = {token} is outside [0, 1]")
            elif not _SIX_DECIMALS.match(token):
                add(WARNING, "label_decimals", i, f"{name} = {token} does not have six decimal places")
            values.append(value)
        else:
            cx, cy, w, h = values
            if w <= 0 or h <= 0:
                add(WARNING, "label_empty_box", i, "has zero width or height")
            elif (cx - w / 2 < -_EDGE_TOLERANCE or cx + w / 2 > 1 + _EDGE_TOLERANCE
                  or cy - h / 2 < -_EDGE_TOLERANCE or cy + h / 2 > 1 + _EDGE_TOLERANCE):
                add(WARNING, "label_not_clipped", i, "box extends past the image; label boxes are clipped (section 8.2)")
    return problems


class LabelFormatError(ValueError):
    pass


def parse_labels(text: str) -> list[Box]:
    """Boxes of a label file's text, in line order. Raises LabelFormatError on any error (warnings pass)."""
    errors = [p for p in check_label_text(text) if p.severity == ERROR]
    if errors:
        raise LabelFormatError("; ".join(f"line {p.line}: {p.message}" for p in errors))
    boxes = []
    for line in text.replace("\r\n", "\n").splitlines():
        class_id, cx, cy, w, h = line.split()
        boxes.append(Box(float(cx), float(cy), float(w), float(h), int(class_id)))
    return boxes


def count_label_lines(text: str) -> int:
    return len(text.replace("\r\n", "\n").splitlines())


def format_labels(boxes: list[Box]) -> str:
    """The exact file text: six decimals, single spaces, final newline; empty for no cars."""
    for b in boxes:
        if b.class_id not in CLASS_IDS:
            raise ValueError(f"class_id {b.class_id} is not a class of the standard")
        for name in ("cx", "cy", "w", "h"):
            value = getattr(b, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} = {value} is outside [0, 1]")
    return "".join(f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


def read_labels(path: str | os.PathLike) -> list[Box]:
    return parse_labels(Path(path).read_text(encoding="utf-8"))


def write_labels(path: str | os.PathLike, boxes: list[Box], *, overwrite: bool = True) -> Path:
    """Write a label file. Keep tracks.json in sync: its entries refer to these lines by index."""
    return write_text(path, format_labels(boxes), overwrite=overwrite)
