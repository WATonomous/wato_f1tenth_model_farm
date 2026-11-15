"""Builds small, fully valid sessions for tests (real PNGs, every file the standard asks for)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from farm.standard import images, jsonio, labels, provenance, schema
from farm.standard.layout import SessionDir

SESSION_ID = "20261115_e7-hallway_r01"
W, H = 32, 24
TOOL = "sam3-autolabel@0.1.0"
MEMBER = "muhtasim001"


def session_doc(session_id: str = SESSION_ID, width: int = W, height: int = H) -> dict:
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "environment": "hallway",
        "location": session_id.split("_")[1],
        "recorded_start_ns": 1_800_000_000_000_000_000,
        "recorded_end_ns": 1_800_000_060_000_000_000,
        "source": {
            "bag_path": f"recordings/{session_id}",
            "storage": "mcap",
            "rgb_topic": "/camera/camera/color/image_raw/compressed",
            "rgb_source_encoding": "jpeg",
            "rgb_info_topic": "/camera/camera/color/camera_info",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw/zstd",
            "depth_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
        },
        "camera": {"model": "Intel RealSense D435i", "serial": None, "firmware": None},
        "image": {"width": width, "height": height},
        "depth": {"units": "mm", "invalid_value": 0, "source_depth_scale_m": 0.001,
                  "aligned_to_rgb": True, "aligned_by": "realsense_driver"},
        "calibration": {
            "source": "camera_info",
            "rgb_intrinsics": {"fx": 30.0, "fy": 30.0, "cx": 16.0, "cy": 12.0},
            "distortion_model": "plumb_bob",
            "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "depth_to_rgb": {"rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                             "translation_m": [0.015, 0.0, 0.0]},
            "rgb_to_base_link": None,
        },
        "pairing": {"max_dt_ns": 10_000_000, "rgb_frames": 4, "paired": 3, "rejected": 1},
        "extraction": {"tool": "extract-pairs@0.1.0", "date": "2026-11-15T09:00:00Z", "stride": 1},
    }


def build_session(root: Path, session_id: str = SESSION_ID) -> SessionDir:
    """Three samples:

    000000  detection generated (2 boxes, one truncated), segmentation generated (mask)
    000001  nothing annotated (no label, no mask)
    000002  detection excluded (no label), segmentation corrected by hand (mask)
    """
    s = SessionDir(root / session_id)
    jsonio.write_json(s.session_json, session_doc(session_id))
    rng = np.random.default_rng(0)
    tracks = schema.new_tracks(session_id)
    tracks["tracks"] = {"1": {"subtype": "f1tenth", "note": None}, "2": {"subtype": "rc_car", "note": None}}
    base_ts = 1_800_000_001_000_000_000
    for i in range(3):
        rgb_ts = base_ts + i * 33_333_333
        depth = rng.integers(0, 4000, size=(H, W), dtype=np.uint16)
        meta = schema.new_meta(session_id, i, rgb_ts, rgb_ts + 1_500_000, W, H,
                               images.depth_valid_fraction(depth))
        sid = meta["sample_id"]
        images.write_rgb(s.file("rgb", sid), rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
        images.write_depth(s.file("depth", sid), depth)
        det, seg = meta["annotations"]["detection"], meta["annotations"]["segmentation"]
        if i == 0:
            a, _ = labels.Box.from_xyxy_px(4, 6, 14, 14, width=W, height=H)
            b, truncated = labels.Box.from_xyxy_px(-3, 10, 6, 18, width=W, height=H)
            assert truncated
            labels.write_labels(s.file("labels", sid), [a, b])
            tracks["boxes"][sid] = [
                {"line": 0, "track_id": 1, "occluded": False, "truncated": False, "unclipped_xyxy_px": None},
                {"line": 1, "track_id": 2, "occluded": False, "truncated": True, "unclipped_xyxy_px": [-3.0, 10.0, 6.0, 18.0]},
            ]
            det = provenance.apply(det, "generated", TOOL, date="2026-11-15T10:00:00Z")
            seg = provenance.apply(seg, "generated", TOOL, date="2026-11-15T10:00:00Z")
            images.write_mask(s.file("masks", sid), _mask())
        elif i == 2:
            det = provenance.apply(det, "excluded", MEMBER, date="2026-11-15T11:00:00Z",
                                   exclusion_reason="motion_blur", exclusion_note=None)
            seg = provenance.apply(seg, "corrected", MEMBER, date="2026-11-15T11:00:00Z", note="hand-labelled")
            images.write_mask(s.file("masks", sid), _mask())
        meta["annotations"] = {"detection": det, "segmentation": seg}
        jsonio.write_json(s.file("meta", sid), meta)
    jsonio.write_json(s.tracks_json, tracks)
    return s


def _mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[12:, :] = images.DRIVABLE
    mask[:4, :] = images.IGNORE
    return mask


def sample(i: int, session_id: str = SESSION_ID) -> str:
    return f"{session_id}_{i:06d}"
