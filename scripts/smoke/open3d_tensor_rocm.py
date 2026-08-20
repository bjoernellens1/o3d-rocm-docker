#!/usr/bin/env python3
"""Qualify AMD Open3D HIP core Tensor + RGB-D odometry on a real GPU.

AMD's Open3D HIP port deliberately reuses the CUDA-facing Open3D device API,
so an AMD GPU is expected to appear as ``CUDA:0`` to Open3D while the build
itself is backed by HIP/ROCm.
"""
from __future__ import annotations

import json
import numpy as np
import open3d as o3d


def main() -> None:
    assert o3d.core.cuda.is_available(), (
        "Open3D GPU backend unavailable; AMD HIP build should expose the CUDA-facing API"
    )
    device = o3d.core.Device("CUDA:0")

    # Core Tensor allocation, arithmetic and CPU round trip.
    x = o3d.core.Tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=o3d.core.Dtype.Float32,
        device=device,
    )
    y = (x * 2.0 + 1.0).sum()
    got = float(y.cpu().numpy())
    assert abs(got - 24.0) < 1e-5, got
    assert str(x.device) == "CUDA:0", x.device

    # Match the exact Tensor geometry surface used by VarSplat and SGAD-SLAM.
    h = w = 32
    color_np = np.zeros((h, w, 3), dtype=np.float32)
    color_np[..., 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    color_np[..., 1] = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    depth_np = np.full((h, w), 1.5, dtype=np.float32)

    color = o3d.t.geometry.Image(color_np).to(device)
    depth = o3d.t.geometry.Image(depth_np).to(device)
    source = o3d.t.geometry.RGBDImage(color, depth)
    target = source.clone()
    assert str(source.color.device) == "CUDA:0", source.color.device
    assert str(source.depth.device) == "CUDA:0", source.depth.device

    intrinsics = o3d.core.Tensor(
        [[40.0, 0.0, (w - 1) / 2.0], [0.0, 40.0, (h - 1) / 2.0], [0.0, 0.0, 1.0]],
        o3d.core.Dtype.Float64,
    )
    init = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float64)
    criteria = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5),
    ]
    result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        source,
        target,
        intrinsics,
        init,
        1.0,
        10.0,
        criteria,
        o3d.t.pipelines.odometry.Method.Hybrid,
    )
    transform = result.transformation.cpu().numpy()
    assert np.isfinite(transform).all(), transform
    assert transform.shape == (4, 4), transform.shape

    payload = {
        "open3d_version": o3d.__version__,
        "open3d_device": str(device),
        "tensor_sum": got,
        "odometry_fitness": float(getattr(result, "fitness", float("nan"))),
        "odometry_inlier_rmse": float(getattr(result, "inlier_rmse", float("nan"))),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
