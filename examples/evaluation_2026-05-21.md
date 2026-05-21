# Evaluation 2026-05-21

Host:

- Fedora rootless Podman `5.8.2`.
- AMD GPU device passthrough available through `/dev/kfd` and `/dev/dri`.
- User is in `video` and `render`.

Dataset:

- `/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd`
- Checked extracted sequence: `freiburg1_xyz/rgbd_dataset_freiburg1_xyz`

Results:

- Upstream ROCm base container `docker.io/rocm/dev-ubuntu-24.04:7.2` sees the GPU with `rocm-smi --showproductname`: `AMD Radeon 8060S`, `gfx1151`.
- A native HIP vector-add smoke test compiled and ran successfully in the upstream ROCm base container.
- Repo image `localhost/o3d-rocm:local` imports Open3D `0.19.0+1e7b174`.
- `sycl-ls --verbose` loads `libur_adapter_hip.so.0` and lists `[hip:gpu] Radeon 8060S Graphics gfx1151`.
- `open3d.core.sycl.get_available_devices()` returns `[SYCL:0]`.
- The Dockerfile now uses `SYCL_BACKEND=source` by default, pins the Unified Runtime checkout, and applies a HIP device-info patch. That removed the `Unsupported ParamName in urDeviceGetInfo` noise from `sycl-ls --verbose`.
- Open3D tensor allocation/kernel execution on `SYCL:0` fails with `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)`.
- A narrow C++ SYCL probe can create a queue and allocate device memory, but fails on host-to-device copy with `UR_RESULT_ERROR_INVALID_ENUMERATION`.
- `SYCL_UR_TRACE=2` shows the failing call is `urEnqueueUSMMemcpy`.
- `MODE=capability DEVICE=SYCL:0 ./examples/run_tum_examples_podman.sh` confirms the real support boundary: SYCL discovery and `Tensor.empty` pass; host copies, fill kernels, elementwise kernels, linalg, tensor image copies, tensor ICP, RGB-D odometry, and dense SLAM fail before useful work. `VoxelBlockGrid` fails separately with `DeviceHashBackend.cpp:38: Unimplemented device`.
- `MODE=capability DEVICE=CPU:0 ./examples/run_tum_examples_podman.sh` passes the same ladder on CPU.
- Tensor RGB-D odometry on `CPU:0` works on TUM `freiburg1_xyz`; a 4-edge smoke run produced fitness values around `0.65` to `0.68`.
- Tensor Dense RGB-D SLAM on `CPU:0` works on TUM `freiburg1_xyz`; a 20-frame run produced a point cloud with `44047` points in about `1.4 s`.
- Low-level `VoxelBlockGrid` SLAM on `CPU:0` works on TUM `freiburg1_xyz`; the example follows the integrate, ray-cast, and RGB-D odometry loop directly.
- The examples now emit CPU comparison metrics: odometry `edges_per_second`, SLAM `frames_per_second`, tracked-edge rate, mean/median frame time, mean/median tracking time, and VBG integrate/raycast timing.
- The benchmark runner records `CPU:0` and `SYCL:0` results side by side under `examples/output/benchmark/`.
- Recorded 20-frame CPU baseline on `freiburg1_xyz` after the backend/Dockerfile update: tensor RGB-D odometry `45.7368 edges/s`, dense SLAM `20.1332 frames/s`, VBG SLAM `20.0063 frames/s`.
- Tensor RGB-D odometry and Dense RGB-D SLAM on `SYCL:0` fail before useful work because basic Open3D tensor creation on the HIP SYCL backend fails.
- VBG SLAM on `SYCL:0` fails at VoxelBlockGrid construction with `DeviceHashBackend.cpp:38: Unimplemented device`.
- Source inspection shows Open3D v0.19 RGB-D odometry dispatches only CPU and CUDA kernels, and the hash-map backend used by `VoxelBlockGrid` also dispatches only CPU and CUDA.
- Open3D/Embree SYCL build warnings such as `Undefined function intel_has_committed_hit` and `Undefined function intel_ray_query_abandon` apply to the raycasting path. They are not the first tensor blocker, but they support treating `RaycastingScene` as an unreliable AMD ROCm target in this build.

Conclusion:

The upstream ROCm container path and AMD GPU passthrough are working. The local Open3D build exposes the AMD GPU through SYCL discovery, but ROCm/SYCL acceleration for Open3D tensor workloads is not currently working on this stack. The tensor odometry and Dense SLAM examples are valid on CPU and act as focused failure reproducers for `SYCL:0`.
