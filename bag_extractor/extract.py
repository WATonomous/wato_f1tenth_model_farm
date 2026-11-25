"""Export compressed RGB and Zstd depth frames from an offline ROS 2 bag."""

import argparse
import csv
import gzip
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import zstandard

from pair_frames import pair_frames


def decode_depth(data):
    """Read the zstd_image_transport header, including row stride and endianness."""
    if len(data) < 17:
        raise ValueError("Truncated Zstd image header")
    height, width, bigendian, step, encoding_size = struct.unpack_from("<IIBII", data)
    offset = 17 + encoding_size
    if offset >= len(data) or not width or not height:
        raise ValueError("Invalid Zstd image dimensions or encoding length")
    encoding = data[17:offset].decode("utf-8")
    if encoding != "16UC1":
        raise ValueError(f"Expected 16UC1 depth, received {encoding!r}")
    if step < width * 2 or bigendian not in (0, 1):
        raise ValueError("Invalid depth row stride or byte order")
    expected_size = height * step
    payload = io.BytesIO(data[offset:])
    if data[offset:offset + 2] == b"\x1f\x8b":
        stream = gzip.GzipFile(fileobj=payload)
    else:
        stream = zstandard.ZstdDecompressor().stream_reader(payload)
    with stream:
        raw = stream.read(expected_size + 1)
    if len(raw) != expected_size:
        raise ValueError("Decompressed depth size does not match its header")
    dtype = np.dtype(">u2" if bigendian else "<u2")
    pixels = np.ndarray((height, width), dtype=dtype, buffer=raw, strides=(step, 2))
    return pixels.astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="MCAP file or ROS 2 bag directory")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw/zstd")
    parser.add_argument("--limit", type=int, help="Maximum frames per stream")
    parser.add_argument("--max-dt-ns", type=int, default=10_000_000,
                        help="Maximum RGB/depth capture time difference (default 10 ms)")
    args = parser.parse_args()
    if args.max_dt_ns < 0:
        parser.error("--max-dt-ns must be nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    counts = {"rgb": 0, "depth": 0}
    topics = {args.rgb_topic: "rgb", args.depth_topic: "depth"}
    typestore = get_typestore(Stores.ROS2_JAZZY)
    with Reader(args.bag) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for topic in topics:
            count = sum(c.msgcount for c in connections if c.topic == topic)
            print(f"{topic}: {count} recorded messages")
        if not any(c.msgcount for c in connections):
            parser.error("No messages found on either requested topic")
        args.output.mkdir(parents=True, exist_ok=False)
        for kind in counts:
            (args.output / kind).mkdir()
        with (args.output / "frames.csv").open("w", newline="") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(["stream", "file", "capture_timestamp_ns", "bag_timestamp_ns", "frame_id", "format"])
            for connection, bag_time, raw in reader.messages(connections=connections):
                kind = topics[connection.topic]
                if args.limit is not None and counts[kind] >= args.limit:
                    continue
                if connection.msgtype != "sensor_msgs/msg/CompressedImage":
                    raise ValueError(f"Unexpected type {connection.msgtype} on {connection.topic}")
                message = typestore.deserialize_cdr(raw, connection.msgtype)
                stamp = message.header.stamp
                capture_time = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                relative_path = Path(kind) / f"{capture_time}_{counts[kind]:06d}.png"
                data = bytes(message.data)
                if kind == "rgb":
                    with Image.open(io.BytesIO(data)) as image:
                        image.convert("RGB").save(args.output / relative_path)
                else:
                    Image.fromarray(decode_depth(data)).save(args.output / relative_path)
                writer.writerow([kind, relative_path.as_posix(), capture_time, bag_time, message.header.frame_id, message.format])
                counts[kind] += 1
        print(f"Exported {counts['rgb']} RGB and {counts['depth']} depth frames to {args.output}")

    pair_frames(args.output, args.max_dt_ns)


if __name__ == "__main__":
    main()
