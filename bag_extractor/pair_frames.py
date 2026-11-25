"""Pair exported RGB/depth frames by nearest capture time with unique depth use."""

import argparse
from bisect import bisect_left
import csv
from pathlib import Path


def pair_frames(folder, max_dt_ns=10_000_000):
    """Write pairs.csv. A shared depth frame goes to the closest RGB candidate.

    Ties prefer the earlier depth timestamp, then the earlier RGB timestamp,
    then the filename. Rejected RGB frames are not rematched to another depth.
    """
    if max_dt_ns < 0:
        raise ValueError("Pairing tolerance must be nonnegative")
    with (folder / "frames.csv").open(newline="") as source:
        frames = list(csv.DictReader(source))
    streams = {}
    for stream in ("rgb", "depth"):
        streams[stream] = sorted(
            (int(row["capture_timestamp_ns"]), row["file"])
            for row in frames if row["stream"] == stream
        )
    depth_times = [stamp for stamp, _ in streams["depth"]]
    winners = {}
    for rgb_time, rgb_file in streams["rgb"]:
        index = bisect_left(depth_times, rgb_time)
        candidates = streams["depth"][max(0, index - 1):index + 1]
        if not candidates:
            continue
        depth_time, depth_file = min(
            candidates, key=lambda frame: (abs(frame[0] - rgb_time), frame)
        )
        delta = depth_time - rgb_time
        if abs(delta) > max_dt_ns:
            continue
        candidate = (abs(delta), rgb_time, rgb_file, depth_time, depth_file)
        if depth_file not in winners or candidate < winners[depth_file]:
            winners[depth_file] = candidate
    pairs = sorted(winners.values(), key=lambda pair: (pair[1], pair[2]))
    with (folder / "pairs.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "rgb_file", "depth_file", "rgb_timestamp_ns", "depth_timestamp_ns", "pair_dt_ns"
        ])
        for _, rgb_time, rgb_file, depth_time, depth_file in pairs:
            writer.writerow([rgb_file, depth_file, rgb_time, depth_time, depth_time - rgb_time])
    print(
        f"Paired {len(pairs)} frames within {max_dt_ns / 1_000_000:g} ms; "
        f"unmatched RGB: {len(streams['rgb']) - len(pairs)}, "
        f"unused depth: {len(streams['depth']) - len(pairs)}"
    )
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Existing extraction directory with frames.csv")
    parser.add_argument("--max-dt-ns", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.max_dt_ns < 0:
        parser.error("--max-dt-ns must be nonnegative")
    pair_frames(args.folder, args.max_dt_ns)
