#!/usr/bin/env python3
"""Low-level VoxelBlockGrid RGB-D SLAM example on a TUM RGB-D sequence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import numpy.typing as npt
import open3d as o3d
from open3d.t.geometry import Image, RGBDImage, VoxelBlockGrid
from open3d.t.pipelines.odometry import (
    Method,
    OdometryConvergenceCriteria,
    OdometryLossParams,
    rgbd_odometry_multi_scale,
)

from tum_tensor_rgbd_odometry import associate_rgb_depth, make_intrinsic


class VBGSLAM:
    def __init__(
        self,
        depth_scale: float = 5000.0,
        depth_max: float = 3.0,
        odometry_max_iter: list[int] | None = None,
        voxel_size: float = 0.0058,
        block_resolution: int = 16,
        block_count: int = 40000,
        device: str | None = None,
    ):
        if device is None:
            device = "SYCL:0" if o3d.core.sycl.is_available() else "CPU:0"
        self.device = o3d.core.Device(device)

        self.depth_scale = depth_scale
        self.depth_min = 0.1
        self.depth_max = depth_max
        self.trunc_voxel_multiplier = 8.0

        if odometry_max_iter is None:
            odometry_max_iter = [6, 3, 1]
        self.odom_criteria_list = [
            OdometryConvergenceCriteria(max_iteration=max_iter) for max_iter in odometry_max_iter
        ]
        self.odom_loss_params = OdometryLossParams()

        self.vg = VoxelBlockGrid(
            attr_names=["tsdf", "weight", "color"],
            attr_dtypes=[
                o3d.core.Dtype.Float32,
                o3d.core.Dtype.UInt16,
                o3d.core.Dtype.UInt16,
            ],
            attr_channels=[1, 1, 3],
            voxel_size=voxel_size,
            block_resolution=block_resolution,
            block_count=block_count,
            device=self.device,
        )

        self.T_frame_to_model = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64, self.device)
        self.raycast_frame_rgbd: RGBDImage | None = None
        self.frame_id = 0
        self.last_odometry_result: Any | None = None

    def iterate(
        self,
        colour: npt.NDArray[np.uint8],
        depth: npt.NDArray[np.uint16] | npt.NDArray[np.float32],
        intrinsics: npt.ArrayLike,
        init_source_to_target: npt.ArrayLike = np.identity(4),
        T_frame_to_model: npt.ArrayLike | None = None,
    ) -> dict[str, float]:
        timings: dict[str, float] = {}
        assert colour.shape[:2] == depth.shape[:2]
        assert colour.dtype == np.uint8

        intrinsics_np = np.asarray(intrinsics)
        assert intrinsics_np.shape == (3, 3)

        if depth.dtype == np.float32:
            depth = np.nan_to_num(depth)
            depth = np.rint(depth * self.depth_scale).astype(np.uint16)

        assert depth.dtype == np.uint16

        imgc = Image(np.ascontiguousarray(colour)).to(self.device)
        imgd = Image(np.ascontiguousarray(depth).astype(np.uint16)).to(self.device)
        input_frame_rgbd = RGBDImage(imgc, imgd)
        intrinsic = o3d.core.Tensor(intrinsics_np, o3d.core.Dtype.Float64, self.device)

        self.frame_id += 1
        self.last_odometry_result = None

        track_t0 = time.time()
        if T_frame_to_model is None:
            if self.raycast_frame_rgbd is not None:
                result = rgbd_odometry_multi_scale(
                    source=input_frame_rgbd,
                    target=self.raycast_frame_rgbd,
                    intrinsics=intrinsic,
                    init_source_to_target=o3d.core.Tensor(
                        init_source_to_target, o3d.core.Dtype.Float64, self.device
                    ),
                    depth_scale=self.depth_scale,
                    depth_max=self.depth_max,
                    params=self.odom_loss_params,
                    criteria_list=self.odom_criteria_list,
                    method=Method.PointToPlane,
                )
                self.T_frame_to_model = self.T_frame_to_model @ result.transformation
                self.last_odometry_result = result
        else:
            self.T_frame_to_model = o3d.core.Tensor(
                T_frame_to_model, o3d.core.Dtype.Float64, self.device
            )
        timings["track_s"] = time.time() - track_t0

        extrinsic = self.T_frame_to_model.inv().contiguous()
        blocks_t0 = time.time()
        frustum_block_coords = self.vg.compute_unique_block_coordinates(
            depth=input_frame_rgbd.depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=self.depth_scale,
            depth_max=self.depth_max,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        timings["block_coords_s"] = time.time() - blocks_t0

        integrate_t0 = time.time()
        self.vg.integrate(
            block_coords=frustum_block_coords,
            depth=input_frame_rgbd.depth,
            color=input_frame_rgbd.color,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            depth_scale=self.depth_scale,
            depth_max=self.depth_max,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        timings["integrate_s"] = time.time() - integrate_t0

        weight_threshold = min(self.frame_id * 1.0, 3.0)
        raycast_t0 = time.time()
        rendered_attributes = self.vg.ray_cast(
            block_coords=frustum_block_coords,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=input_frame_rgbd.color.columns,
            height=input_frame_rgbd.color.rows,
            render_attributes=["depth", "color"],
            depth_scale=self.depth_scale,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            weight_threshold=weight_threshold,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        timings["raycast_s"] = time.time() - raycast_t0

        self.raycast_frame_rgbd = RGBDImage(
            Image(rendered_attributes["color"]),
            Image(rendered_attributes["depth"]),
        )
        return timings

    def get_pose(self) -> npt.NDArray[np.float64]:
        return self.T_frame_to_model.cpu().numpy()

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        surface_weight_thr = 3.0
        if self.vg.hashmap().active_buf_indices().shape[0] == 0:
            return o3d.geometry.TriangleMesh()
        return self.vg.extract_triangle_mesh(weight_threshold=surface_weight_thr).to_legacy()

    def get_points(self, weight_threshold: float = 3.0) -> o3d.t.geometry.PointCloud:
        if self.vg.hashmap().active_buf_indices().shape[0] == 0:
            return o3d.t.geometry.PointCloud(self.device)
        return self.vg.extract_point_cloud(weight_threshold=weight_threshold)


def read_legacy_image(path: Path) -> npt.NDArray[Any]:
    return np.asarray(o3d.io.read_image(str(path)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--device", default="CPU:0")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--association-window", type=float, default=0.02)
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.0058)
    parser.add_argument("--block-count", type=int, default=40000)
    parser.add_argument("--pointcloud-out", type=Path)
    parser.add_argument("--mesh-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sequence_dir = args.sequence.resolve()
    pairs = associate_rgb_depth(sequence_dir, args.association_window)
    if not pairs:
        raise RuntimeError(f"No associated RGB-D frames found in {sequence_dir}")

    intrinsic_tensor = make_intrinsic(sequence_dir, o3d.core.Device("CPU:0"))
    intrinsic_np = intrinsic_tensor.numpy()
    slam = VBGSLAM(
        depth_scale=args.depth_scale,
        depth_max=args.depth_max,
        voxel_size=args.voxel_size,
        block_count=args.block_count,
        device=args.device,
    )

    frame_limit = min(args.max_frames, len(pairs))
    stats: list[dict[str, Any]] = []
    t0 = time.time()

    for i, (rgb_path, depth_path, _, _) in enumerate(pairs[:frame_limit]):
        frame_t0 = time.time()
        timings = slam.iterate(
            colour=read_legacy_image(rgb_path),
            depth=read_legacy_image(depth_path),
            intrinsics=intrinsic_np,
        )
        pose = slam.get_pose()
        row: dict[str, Any] = {
            "frame": i,
            "frame_elapsed_s": time.time() - frame_t0,
            **timings,
            "translation_norm": float(np.linalg.norm(pose[:3, 3])),
        }
        if slam.last_odometry_result is not None:
            row["fitness"] = float(slam.last_odometry_result.fitness)
            row["inlier_rmse"] = float(slam.last_odometry_result.inlier_rmse)
        stats.append(row)

    loop_elapsed_s = time.time() - t0
    extract_t0 = time.time()
    point_count = 0
    points = slam.get_points().cpu()
    if "positions" in points.point:
        point_count = int(points.point.positions.shape[0])
    if args.pointcloud_out:
        args.pointcloud_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.t.io.write_point_cloud(str(args.pointcloud_out), points)
    extract_pointcloud_s = time.time() - extract_t0

    mesh_vertices = 0
    extract_mesh_s = 0.0
    if args.mesh_out:
        mesh_t0 = time.time()
        args.mesh_out.parent.mkdir(parents=True, exist_ok=True)
        mesh = slam.get_mesh()
        mesh_vertices = len(mesh.vertices)
        o3d.io.write_triangle_mesh(str(args.mesh_out), mesh)
        extract_mesh_s = time.time() - mesh_t0

    tracked = [s for s in stats if "fitness" in s]
    total_elapsed_s = time.time() - t0
    frame_times = [s["frame_elapsed_s"] for s in stats]
    track_times = [s["track_s"] for s in tracked]
    integrate_times = [s["integrate_s"] for s in stats]
    raycast_times = [s["raycast_s"] for s in stats]
    summary: dict[str, Any] = {
        "open3d_version": o3d.__version__,
        "sequence": str(sequence_dir),
        "device": args.device,
        "associated_pairs": len(pairs),
        "processed_frames": frame_limit,
        "tracked_edges": len(tracked),
        "elapsed_s": loop_elapsed_s,
        "loop_elapsed_s": loop_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "extract_pointcloud_s": extract_pointcloud_s,
        "extract_mesh_s": extract_mesh_s,
        "frames_per_second": frame_limit / loop_elapsed_s if loop_elapsed_s > 0 else 0.0,
        "tracked_edges_per_second": len(tracked) / loop_elapsed_s if loop_elapsed_s > 0 else 0.0,
        "mean_frame_s": mean(frame_times) if frame_times else 0.0,
        "median_frame_s": median(frame_times) if frame_times else 0.0,
        "mean_track_s": mean(track_times) if track_times else 0.0,
        "median_track_s": median(track_times) if track_times else 0.0,
        "mean_integrate_s": mean(integrate_times) if integrate_times else 0.0,
        "median_integrate_s": median(integrate_times) if integrate_times else 0.0,
        "mean_raycast_s": mean(raycast_times) if raycast_times else 0.0,
        "median_raycast_s": median(raycast_times) if raycast_times else 0.0,
        "mean_fitness": float(np.mean([s["fitness"] for s in tracked])) if tracked else 0.0,
        "mean_inlier_rmse": float(np.mean([s["inlier_rmse"] for s in tracked])) if tracked else 0.0,
        "last_translation_norm": stats[-1]["translation_norm"] if stats else 0.0,
        "pointcloud_points": point_count,
        "mesh_vertices": mesh_vertices,
        "frames": stats,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
