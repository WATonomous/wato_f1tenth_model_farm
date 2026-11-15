"""RGB, depth, mask and probability PNGs (Data Standard, sections 5, 9.1 and 11.3).

Headers are read straight from the PNG's IHDR chunk, so checking a file's size, bit depth and
colour type does not decode the image.

    >>> import numpy as np
    >>> from farm.standard import images
    >>> images.write_depth("d.png", np.zeros((480, 640), dtype=np.uint16))
    >>> images.png_header("d.png")
    PngHeader(width=640, height=480, bit_depth=16, color_type=0)
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .jsonio import write_bytes

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {0: "greyscale", 2: "RGB", 3: "palette", 4: "greyscale+alpha", 6: "RGBA"}

# Mask values (section 9.1).
NOT_DRIVABLE = 0
DRIVABLE = 1
IGNORE = 255
MASK_VALUES = frozenset({NOT_DRIVABLE, DRIVABLE, IGNORE})
PREDICTION_MASK_VALUES = frozenset({NOT_DRIVABLE, DRIVABLE})  # predictions never contain 255


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int

    @property
    def kind(self) -> str:
        return f"{self.bit_depth}-bit {COLOR_TYPES.get(self.color_type, f'colour type {self.color_type}')}"


def png_header(path: str | os.PathLike) -> PngHeader:
    """Read the IHDR chunk; raises ValueError if the file is not a PNG."""
    with open(path, "rb") as f:
        head = f.read(33)
    if len(head) < 33 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a PNG file")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", head[16:26])
    return PngHeader(width, height, bit_depth, color_type)


def _write_png(path: str | os.PathLike, image: Image.Image, overwrite: bool) -> Path:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return write_bytes(path, buf.getvalue(), overwrite=overwrite)


def _check_shape(array: np.ndarray, ndim: int, what: str) -> None:
    if array.ndim != ndim or (ndim == 3 and array.shape[2] != 3):
        raise ValueError(f"{what} must have shape {'(H, W, 3)' if ndim == 3 else '(H, W)'}, got {array.shape}")


# ---- RGB (section 5.1): 8-bit, 3 channels, RGB order

def read_rgb(path: str | os.PathLike) -> np.ndarray:
    """(H, W, 3) uint8 in RGB order. (OpenCV would give BGR; this does not.)"""
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"{path}: RGB image must be 8-bit RGB, got mode {im.mode}")
        return np.asarray(im, dtype=np.uint8).copy()


def write_rgb(path: str | os.PathLike, rgb: np.ndarray, *, overwrite: bool = True) -> Path:
    rgb = np.asarray(rgb)
    _check_shape(rgb, 3, "RGB image")
    if rgb.dtype != np.uint8:
        raise ValueError(f"RGB image must be uint8, got {rgb.dtype}")
    return _write_png(path, Image.fromarray(rgb), overwrite)


# ---- Depth (section 5.2): 16-bit, 1 channel, millimetres, 0 = no return

def read_depth(path: str | os.PathLike) -> np.ndarray:
    """(H, W) uint16 depth in millimetres."""
    header = png_header(path)
    if header.bit_depth != 16 or header.color_type != 0:
        raise ValueError(f"{path}: depth must be a 16-bit greyscale PNG, got {header.kind}")
    with Image.open(path) as im:
        return np.asarray(im).astype(np.uint16)


def write_depth(path: str | os.PathLike, depth_mm: np.ndarray, *, overwrite: bool = True) -> Path:
    depth_mm = np.asarray(depth_mm)
    _check_shape(depth_mm, 2, "depth image")
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"depth must be uint16 millimetres, got {depth_mm.dtype} (convert before writing)")
    return _write_png(path, Image.fromarray(depth_mm), overwrite)


def depth_valid_fraction(depth_mm: np.ndarray, invalid_value: int = 0) -> float:
    """meta depth.valid_fraction: the share of pixels that are not invalid_value."""
    depth_mm = np.asarray(depth_mm)
    return float(np.count_nonzero(depth_mm != invalid_value)) / depth_mm.size if depth_mm.size else 0.0


# ---- Masks (sections 9.1 and 11.3): 8-bit greyscale (mode L), values 0/1/255

def read_mask(path: str | os.PathLike) -> np.ndarray:
    """(H, W) uint8. Raises if the file is not 8-bit greyscale (a palette or RGB PNG is not a mask)."""
    header = png_header(path)
    if header.bit_depth != 8 or header.color_type != 0:
        raise ValueError(f"{path}: mask must be an 8-bit greyscale PNG (mode L), got {header.kind}")
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.uint8).copy()


def bad_mask_values(mask: np.ndarray, allowed: frozenset[int] = MASK_VALUES) -> list[int]:
    return sorted(int(v) for v in np.unique(mask) if int(v) not in allowed)


def write_mask(path: str | os.PathLike, mask: np.ndarray, *, prediction: bool = False,
               overwrite: bool = True) -> Path:
    """Write an annotation mask (0/1/255), or with prediction=True a predicted mask (0/1 only)."""
    mask = np.asarray(mask)
    _check_shape(mask, 2, "mask")
    if mask.dtype != np.uint8:
        raise ValueError(f"mask must be uint8, got {mask.dtype}")
    bad = bad_mask_values(mask, PREDICTION_MASK_VALUES if prediction else MASK_VALUES)
    if bad:
        raise ValueError(f"mask holds values the standard does not allow: {bad}")
    return _write_png(path, Image.fromarray(mask), overwrite)


def write_probs(path: str | os.PathLike, p_drivable: np.ndarray, *, overwrite: bool = True) -> Path:
    """Section 11.3 probs/: value = round(255 * P(drivable)); 255 here means probability 1."""
    p = np.asarray(p_drivable, dtype=np.float64)
    _check_shape(p, 2, "probability map")
    if not np.isfinite(p).all() or p.min(initial=0.0) < 0 or p.max(initial=0.0) > 1:
        raise ValueError("probabilities must be finite and in [0, 1]")
    return _write_png(path, Image.fromarray(np.rint(255 * p).astype(np.uint8)), overwrite)
