# GPU Failure Analysis 2026-05-21

Question:

- Why does `SYCL:0` show up, while Open3D tensor odometry and SLAM do not run on the AMD GPU?

Findings:

- ROCm passthrough is not the blocker. The upstream base image `docker.io/rocm/dev-ubuntu-24.04:7.2` sees the AMD Radeon 8060S / `gfx1151`, and a native HIP vector-add compiled and ran on the GPU.
- SYCL discovery is not the blocker. `sycl-ls --verbose` loads `libur_adapter_hip.so.0`, and Open3D reports `[SYCL:0]`.
- The Dockerfile now has explicit backend modes. The default `SYCL_BACKEND=source` pins the Unified Runtime source checkout and applies a small HIP device-info patch so optional device-aspect queries no longer pollute `sycl-ls --verbose` with unsupported enumeration errors. `SYCL_BACKEND=none` builds CPU-only Open3D, and `SYCL_BACKEND=codeplay` requires an explicit Codeplay plugin `.deb` URL.
- The low-level DPC++ HIP backend is not healthy enough for real tensor work. A minimal DPC++ probe creates a GPU queue and allocates device memory, but fails on the first host-to-device copy with `UR_RESULT_ERROR_INVALID_ENUMERATION`.
- Open3D tensor allocation reflects the same problem: `Tensor.empty(..., SYCL:0)` can allocate, but `Tensor.zeros`, `Tensor.ones`, list construction, `Tensor.eye`, and CPU-to-SYCL copies fail with `UR_RESULT_ERROR_INVALID_ENUMERATION`.
- `SYCL_UR_TRACE=2` narrows the first real failure to `urEnqueueUSMMemcpy` returning `UR_RESULT_ERROR_INVALID_ENUMERATION` on a host-to-device copy. The source-built adapter exposes the device, but command submission is still broken on this ROCm 7.2 / `gfx1151` stack.
- `RaycastingScene` is a separate weak point. The Open3D/Embree SYCL build emits warnings about undefined Intel ray query functions such as `intel_has_committed_hit` and `intel_ray_query_abandon`, and the capability probe fails `RaycastingScene` initialization on `SYCL:0`.
- `open3d.t.pipelines.odometry` is not implemented for SYCL in Open3D v0.19. `RGBDOdometry.cpp` dispatches CPU and CUDA only, then logs `Unimplemented device` for other devices.
- `VoxelBlockGrid` is not implemented for SYCL in Open3D v0.19. `DeviceHashBackend.cpp` dispatches CPU and CUDA only, then logs `Unimplemented device`. VBG SLAM depends on this hash map backend.
- `t.pipelines.slam.Model` is built on `VoxelBlockGrid` and calls tensor RGB-D odometry for frame-to-model tracking, so it inherits both limitations.

Conclusion:

There are two independent blockers:

1. The container's SYCL/HIP runtime stack is incomplete or mismatched. It can enumerate `SYCL:0`, but basic copy/fill/kernel submission through DPC++ HIP fails.
2. The Open3D v0.19 RGB-D odometry and sparse VBG SLAM paths are CPU/CUDA implementations, not SYCL implementations.

Implication:

Fixing the HIP adapter/runtime may make simple Open3D tensor ops work on `SYCL:0`, but it will not make Open3D RGB-D odometry or VBG dense SLAM run on ROCm. Those paths need Open3D SYCL kernels/backends or a different GPU implementation path.

Most plausible next checks:

- Replace the locally built Unified Runtime HIP adapter with a version matched to the installed Intel oneAPI runtime, or use the official Codeplay oneAPI plugin for AMD if a ROCm 7.2 / `gfx1151` compatible package becomes available.
- Re-test with `examples/sycl_runtime_probe.cpp` before re-testing Open3D.
- Keep `MODE=capability DEVICE=SYCL:0 ./examples/run_tum_examples_podman.sh` as the first Open3D-level regression test. If tensor copy/fill still fails there, ICP and RGB-D SLAM cannot be meaningfully accelerated.
- Treat Open3D TUM odometry/SLAM as CPU-only in this repo unless Open3D grows SYCL support for RGB-D odometry and VBG/hashmap kernels.
