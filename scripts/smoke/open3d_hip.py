#!/usr/bin/env python3
"""Exercise the Open3D GPU Tensor and RGB-D odometry APIs on ROCm."""
from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d
import torch

assert os.environ.get("EXPECT_BACKEND") == "rocm"
assert torch.cuda.is_available(), "PyTorch cannot see the ROCm GPU"
assert torch.version.hip, "PyTorch is not using a ROCm/HIP backend"
assert o3d.core.cuda.is_available(), "Open3D HIP GPU module is unavailable"

device = o3d.core.Device("CUDA:0")
x = o3d.core.Tensor([[1.0, 2.0], [3.0, 4.0]], o3d.core.Dtype.Float32, device)
y = (x * 2.0 + 1.0).sum().cpu().numpy()
assert abs(float(y) - 24.0) < 1e-5

h = w = 32
rgb = np.zeros((h, w, 3), dtype=np.float32)
rgb[..., 0] = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
rgb[..., 1] = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
depth = np.full((h, w), 1.5, dtype=np.float32)
source = o3d.t.geometry.RGBDImage(
    o3d.t.geometry.Image(rgb).to(device),
    o3d.t.geometry.Image(depth).to(device),
)
target = source.clone()
intrinsics = o3d.core.Tensor(
    [[40.0, 0.0, 15.5], [0.0, 40.0, 15.5], [0.0, 0.0, 1.0]],
    o3d.core.Dtype.Float64,
)
criteria = [o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5) for _ in range(3)]
result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
    source,
    target,
    intrinsics,
    o3d.core.Tensor(np.eye(4)),
    1.0,
    10.0,
    criteria,
    o3d.t.pipelines.odometry.Method.Hybrid,
)
transform = result.transformation.cpu().numpy()
assert np.isfinite(transform).all()

print(
    json.dumps(
        {
            "backend": "rocm",
            "hip": torch.version.hip,
            "torch_device": torch.cuda.get_device_name(0),
            "open3d_version": o3d.__version__,
            "capabilities": [
                "open3d_tensor_gpu",
                "open3d_t_geometry_gpu",
                "open3d_rgbd_odometry_gpu",
            ],
        },
        sort_keys=True,
    )
)
