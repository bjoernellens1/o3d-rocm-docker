#!/usr/bin/env python3
"""Run Open3D tensor RGB-D odometry on an extracted TUM RGB-D sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import open3d as o3d


TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def infer_intrinsics(sequence_dir: Path) -> tuple[float, float, float, float]:
    name = sequence_dir.name.lower()
    for key, values in TUM_INTRINSICS.items():
        if key in name:
            return values
    raise ValueError(f"Cannot infer TUM intrinsics from sequence path: {sequence_dir}")


def read_tum_list(sequence_dir: Path, filename: str) -> list[tuple[float, Path]]:
    items: list[tuple[float, Path]] = []
    for line in (sequence_dir / filename).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relpath = line.split()[:2]
        items.append((float(timestamp), sequence_dir / relpath))
    return items


def associate_rgb_depth(
    sequence_dir: Path, max_delta_s: float
) -> list[tuple[Path, Path, float, float]]:
    rgb = read_tum_list(sequence_dir, "rgb.txt")
    depth = read_tum_list(sequence_dir, "depth.txt")
    pairs: list[tuple[Path, Path, float, float]] = []
    j = 0
    for rgb_ts, rgb_path in rgb:
        while j + 1 < len(depth) and abs(depth[j + 1][0] - rgb_ts) < abs(depth[j][0] - rgb_ts):
            j += 1
        depth_ts, depth_path = depth[j]
        if abs(depth_ts - rgb_ts) <= max_delta_s:
            pairs.append((rgb_path, depth_path, rgb_ts, depth_ts))
    return pairs


def make_intrinsic(sequence_dir: Path, device: o3d.core.Device) -> o3d.core.Tensor:
    fx, fy, cx, cy = infer_intrinsics(sequence_dir)
    return o3d.core.Tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        o3d.core.Dtype.Float64,
        device,
    )


def read_rgbd(rgb_path: Path, depth_path: Path, device: o3d.core.Device) -> o3d.t.geometry.RGBDImage:
    color = o3d.t.io.read_image(str(rgb_path)).to(device)
    depth = o3d.t.io.read_image(str(depth_path)).to(device)
    return o3d.t.geometry.RGBDImage(color, depth)


def odometry_method(name: str) -> o3d.t.pipelines.odometry.Method:
    return {
        "hybrid": o3d.t.pipelines.odometry.Method.Hybrid,
        "intensity": o3d.t.pipelines.odometry.Method.Intensity,
        "point_to_plane": o3d.t.pipelines.odometry.Method.PointToPlane,
    }[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--device", default="CPU:0")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--association-window", type=float, default=0.02)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument(
        "--method",
        choices=("hybrid", "intensity", "point_to_plane"),
        default="hybrid",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    sequence_dir = args.sequence.resolve()
    pairs = associate_rgb_depth(sequence_dir, args.association_window)
    if len(pairs) < 2:
        raise RuntimeError(f"Need at least two associated RGB-D frames in {sequence_dir}")

    device = o3d.core.Device(args.device)
    intrinsic = make_intrinsic(sequence_dir, device)
    identity = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, device)
    pose = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, device)
    criteria = [o3d.t.pipelines.odometry.OdometryConvergenceCriteria(v) for v in (10, 5, 3)]
    method = odometry_method(args.method)

    frame_limit = min(args.max_frames, len(pairs))
    stats: list[dict[str, Any]] = []
    t0 = time.time()
    previous = read_rgbd(pairs[0][0], pairs[0][1], device)
    ok = 0

    for i in range(1, frame_limit):
        frame_t0 = time.time()
        current = read_rgbd(pairs[i][0], pairs[i][1], device)
        try:
            result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                current,
                previous,
                intrinsic,
                identity,
                args.depth_scale,
                args.depth_max,
                criteria,
                method,
            )
        except Exception as exc:  # noqa: BLE001 - keep benchmark output inspectable.
            stats.append(
                {
                    "frame": i,
                    "elapsed_s": time.time() - frame_t0,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if args.fail_fast:
                raise
            break
        pose = pose @ result.transformation
        ok += 1
        stats.append(
            {
                "frame": i,
                "ok": True,
                "elapsed_s": time.time() - frame_t0,
                "fitness": float(result.fitness),
                "inlier_rmse": float(result.inlier_rmse),
                "translation_norm": float(np.linalg.norm(pose.cpu().numpy()[:3, 3])),
            }
        )
        previous = current

    elapsed_s = time.time() - t0
    success_stats = [s for s in stats if s.get("ok")]
    failed_stats = [s for s in stats if not s.get("ok", True)]
    frame_times = [s["elapsed_s"] for s in success_stats]
    summary: dict[str, Any] = {
        "open3d_version": o3d.__version__,
        "sequence": str(sequence_dir),
        "device": args.device,
        "method": args.method,
        "associated_pairs": len(pairs),
        "processed_edges": ok,
        "failed_edges": len(failed_stats),
        "completed_all_requested_edges": ok == frame_limit - 1,
        "elapsed_s": elapsed_s,
        "edges_per_second": ok / elapsed_s if elapsed_s > 0 else 0.0,
        "mean_edge_s": mean(frame_times) if frame_times else 0.0,
        "median_edge_s": median(frame_times) if frame_times else 0.0,
        "mean_fitness": float(np.mean([s["fitness"] for s in success_stats])) if success_stats else 0.0,
        "mean_inlier_rmse": float(np.mean([s["inlier_rmse"] for s in success_stats]))
        if success_stats
        else 0.0,
        "last_translation_norm": next(
            (s["translation_norm"] for s in reversed(success_stats)), 0.0
        ),
        "first_error": failed_stats[0]["error"] if failed_stats else "",
        "frames": stats,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
