# AMD Open3D HIP container

Canonical ResearchFlow candidate for GPU-backed Open3D Tensor operations on AMD.

Pinned inputs:

- AMD Open3D `moat-port`: `50bb2505be991392f7cdfd040802db7bbf6ef2a8`
- MOAT audit revision: `f69bb67d70e7a47af00095dab7029c230a30be73`
- ROCm: 7.2.4
- ROCm base image digest: `sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`
- Python wheel target: CPython 3.10

The build uses AMD Open3D's native `USE_HIP=ON` path, not the earlier
SYCL/Unified-Runtime workaround. AMD deliberately preserves Open3D's CUDA-facing
GPU device surface, so existing baseline code using `o3d.core.Device("CUDA:0")`
can execute on ROCm without a method-specific device-string patch.

A candidate is not qualified until `scripts/smoke/open3d_tensor_rocm.py` runs on
real AMD hardware. The smoke covers core Tensor arithmetic and the exact
`o3d.t.pipelines.odometry.rgbd_odometry_multi_scale` API used by VarSplat and
SGAD-SLAM.

The AMD port is based on Open3D 0.19.0. VarSplat and SGAD-SLAM paper-mode images
pin Open3D 0.18.0; SplaTAM pins 0.16.0. A newer Open3D is therefore an explicit
controlled/stress portability adaptation until a version-specific backport or
cross-version equivalence gate passes.
