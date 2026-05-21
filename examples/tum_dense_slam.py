#!/usr/bin/env python3
"""Run the Open3D tensor Dense RGB-D SLAM loop on a TUM RGB-D sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import open3d as o3d

from tum_tensor_rgbd_odometry import associate_rgb_depth, make_intrinsic


def read_frame(
    rgb_path: Path,
    depth_path: Path,
    height: int,
    width: int,
    intrinsic: o3d.core.Tensor,
    device: o3d.core.Device,
) -> o3d.t.pipelines.slam.Frame:
    frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)
    frame.set_data_from_image("depth", o3d.t.io.read_image(str(depth_path)).to(device))
    frame.set_data_from_image("color", o3d.t.io.read_image(str(rgb_path)).to(device))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--device", default="CPU:0")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--association-window", type=float, default=0.02)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--depth-diff", type=float, default=0.07)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--block-count", type=int, default=2000)
    parser.add_argument("--trunc-voxel-multiplier", type=float, default=8.0)
    parser.add_argument("--pointcloud-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sequence_dir = args.sequence.resolve()
    pairs = associate_rgb_depth(sequence_dir, args.association_window)
    if not pairs:
        raise RuntimeError(f"No associated RGB-D frames found in {sequence_dir}")

    device = o3d.core.Device(args.device)
    intrinsic = make_intrinsic(sequence_dir, device)
    transform = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float64, device)
    model = o3d.t.pipelines.slam.Model(args.voxel_size, 16, args.block_count, transform, device)

    depth_ref = o3d.t.io.read_image(str(pairs[0][1]))
    height, width = depth_ref.rows, depth_ref.columns
    raycast_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)
    criteria = [o3d.t.pipelines.odometry.OdometryConvergenceCriteria(v) for v in (6, 3, 1)]

    frame_limit = min(args.max_frames, len(pairs))
    stats: list[dict[str, Any]] = []
    t0 = time.time()

    for i, (rgb_path, depth_path, _, _) in enumerate(pairs[:frame_limit]):
        frame_t0 = time.time()
        input_frame = read_frame(rgb_path, depth_path, height, width, intrinsic, device)
        track_elapsed_s = 0.0
        if i > 0:
            track_t0 = time.time()
            result = model.track_frame_to_model(
                input_frame,
                raycast_frame,
                args.depth_scale,
                args.depth_max,
                args.depth_diff,
                o3d.t.pipelines.odometry.Method.PointToPlane,
                criteria,
            )
            track_elapsed_s = time.time() - track_t0
            transform = transform @ result.transformation
            stats.append(
                {
                    "frame": i,
                    "track_elapsed_s": track_elapsed_s,
                    "fitness": float(result.fitness),
                    "inlier_rmse": float(result.inlier_rmse),
                    "translation_norm": float(np.linalg.norm(transform.cpu().numpy()[:3, 3])),
                }
            )

        model.update_frame_pose(i, transform)
        model.integrate(input_frame, args.depth_scale, args.depth_max, args.trunc_voxel_multiplier)
        model.synthesize_model_frame(
            raycast_frame,
            args.depth_scale,
            args.depth_min,
            args.depth_max,
            args.trunc_voxel_multiplier,
            False,
        )
        if i == 0:
            stats.append({"frame": i, "track_elapsed_s": 0.0})
        stats[-1]["frame_elapsed_s"] = time.time() - frame_t0

    loop_elapsed_s = time.time() - t0
    extract_t0 = time.time()
    pointcloud_points = 0
    if args.pointcloud_out:
        args.pointcloud_out.parent.mkdir(parents=True, exist_ok=True)
        pointcloud = model.extract_pointcloud(3.0).cpu()
        pointcloud_points = int(pointcloud.point.positions.shape[0])
        o3d.t.io.write_point_cloud(str(args.pointcloud_out), pointcloud)
    else:
        pointcloud = model.extract_pointcloud(3.0)
        pointcloud_points = int(pointcloud.point.positions.shape[0])
    extract_pointcloud_s = time.time() - extract_t0

    total_elapsed_s = time.time() - t0
    frame_times = [s["frame_elapsed_s"] for s in stats if "frame_elapsed_s" in s]
    track_times = [s["track_elapsed_s"] for s in stats if s.get("track_elapsed_s", 0.0) > 0.0]
    summary: dict[str, Any] = {
        "open3d_version": o3d.__version__,
        "sequence": str(sequence_dir),
        "device": args.device,
        "associated_pairs": len(pairs),
        "processed_frames": frame_limit,
        "tracked_edges": len(track_times),
        "elapsed_s": loop_elapsed_s,
        "loop_elapsed_s": loop_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "extract_pointcloud_s": extract_pointcloud_s,
        "frames_per_second": frame_limit / loop_elapsed_s if loop_elapsed_s > 0 else 0.0,
        "tracked_edges_per_second": len(track_times) / loop_elapsed_s if loop_elapsed_s > 0 else 0.0,
        "mean_frame_s": mean(frame_times) if frame_times else 0.0,
        "median_frame_s": median(frame_times) if frame_times else 0.0,
        "mean_track_s": mean(track_times) if track_times else 0.0,
        "median_track_s": median(track_times) if track_times else 0.0,
        "mean_fitness": float(np.mean([s["fitness"] for s in stats if "fitness" in s]))
        if track_times
        else 0.0,
        "mean_inlier_rmse": float(np.mean([s["inlier_rmse"] for s in stats if "inlier_rmse" in s]))
        if track_times
        else 0.0,
        "last_translation_norm": next(
            (s["translation_norm"] for s in reversed(stats) if "translation_norm" in s), 0.0
        ),
        "pointcloud_points": pointcloud_points,
        "frames": stats,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
