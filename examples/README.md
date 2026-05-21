# Open3D ROCm Examples

These examples use Fedora rootless Podman with AMD GPU passthrough:

```bash
--device=/dev/kfd
--device=/dev/dri
--group-add keep-groups
--security-opt=label=disable
--security-opt=seccomp=unconfined
--ipc=host
```

The image built by this repo starts from `docker.io/rocm/dev-ubuntu-24.04:7.2`
and installs Open3D v0.19.0. The default build keeps ROCm 7.2 for `gfx1151`
and treats Open3D SYCL as an experimental probe:

```bash
podman build \
  --build-arg SYCL_BACKEND=source \
  -t localhost/o3d-rocm:local .
```

`SYCL_BACKEND=source` builds a pinned Unified Runtime HIP adapter from source.
`SYCL_BACKEND=none` builds a CPU-only Open3D image. `SYCL_BACKEND=codeplay`
requires `CODEPLAY_AMD_PLUGIN_DEB_URL` and is a separate compatibility check,
not the recommended baseline for ROCm 7.2 / `gfx1151`.

## Validate ROCm Passthrough

Run the upstream ROCm base image directly:

```bash
podman run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add keep-groups \
  --security-opt=label=disable \
  --security-opt=seccomp=unconfined \
  --ipc=host \
  docker.io/rocm/dev-ubuntu-24.04:7.2 \
  rocm-smi --showproductname
```

On the checked Fedora host this reported `AMD Radeon 8060S`, `gfx1151`.

## Open3D SYCL Probe

```bash
MODE=sycl-smoke DEVICE=SYCL:0 ./examples/run_tum_examples_podman.sh
```

On the checked image, Open3D sees `SYCL:0` as the HIP GPU, but a simple Open3D tensor kernel currently fails with `UR_RESULT_ERROR_INVALID_ENUMERATION`. That means ROCm device passthrough and SYCL discovery work, but Open3D tensor execution on the AMD backend is not validated on this stack yet.

For a lower-level DPC++ check independent of Open3D:

```bash
podman run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add keep-groups \
  --security-opt=label=disable \
  --security-opt=seccomp=unconfined \
  --ipc=host \
  --userns=keep-id \
  -v "$(pwd):/workspace:rw" \
  -w /workspace \
  localhost/o3d-rocm:local \
  bash -lc 'icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1151 examples/sycl_runtime_probe.cpp -o /tmp/sycl_runtime_probe && /tmp/sycl_runtime_probe'
```

See `examples/gpu_failure_analysis_2026-05-21.md` for the current root-cause analysis.

## Open3D SYCL Capability Ladder

Run this before trying ICP or RGB-D SLAM on `SYCL:0`. It starts with discovery
and simple tensor operations, then moves through tensor geometry, ICP, RGB-D
odometry, `VoxelBlockGrid`, and dense SLAM model construction.

```bash
MODE=capability DEVICE=SYCL:0 ./examples/run_tum_examples_podman.sh
MODE=capability DEVICE=CPU:0 ./examples/run_tum_examples_podman.sh
```

Recorded outputs from the checked image:

- `examples/output/capability/open3d_sycl_SYCL_0.md`
- `examples/output/capability/open3d_sycl_SYCL_0.json`
- `examples/output/capability/open3d_cpu_CPU_0.md`
- `examples/output/capability/open3d_cpu_CPU_0.json`

Current `SYCL:0` result: SYCL discovery and `Tensor.empty` allocation pass, but
host copies, fill kernels, elementwise kernels, tensor image copies, ICP,
RGB-D odometry, and dense SLAM fail before useful work with
`UR_RESULT_ERROR_INVALID_ENUMERATION`. `VoxelBlockGrid` fails separately with
`DeviceHashBackend.cpp:38: Unimplemented device`.

## Tensor RGB-D Odometry On TUM

CPU tensor odometry:

```bash
MODE=odometry \
DEVICE=CPU:0 \
MAX_FRAMES=50 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

SYCL tensor odometry diagnostic:

```bash
MODE=odometry \
DEVICE=SYCL:0 \
MAX_FRAMES=10 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

The TUM depth scale is set to `5000.0`, and the script infers Freiburg camera intrinsics from the sequence name.

## Dense RGB-D SLAM On TUM

CPU dense SLAM:

```bash
MODE=slam \
DEVICE=CPU:0 \
MAX_FRAMES=50 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

SYCL dense SLAM diagnostic:

```bash
MODE=slam \
DEVICE=SYCL:0 \
MAX_FRAMES=5 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

Outputs are written under `examples/output/`.

## VoxelBlockGrid SLAM On TUM

This is the lower-level `open3d.t.geometry.VoxelBlockGrid` variant: integrate the input RGB-D frame into a TSDF volume, ray-cast a model view, then use tensor RGB-D odometry against that rendered model view.

```bash
MODE=vbg-slam \
DEVICE=CPU:0 \
MAX_FRAMES=50 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

SYCL diagnostic:

```bash
MODE=vbg-slam \
DEVICE=SYCL:0 \
MAX_FRAMES=5 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

## CPU vs ROCm Metrics

Run all three workloads on `CPU:0` and `SYCL:0` and record comparison metrics:

```bash
MODE=benchmark \
MAX_FRAMES=50 \
SEQUENCE=freiburg1_xyz/rgbd_dataset_freiburg1_xyz \
./examples/run_tum_examples_podman.sh
```

The benchmark writes:

- `examples/output/benchmark/summary.json`
- `examples/output/benchmark/summary.md`

For working runs it records throughput fields such as `edges_per_second`, `frames_per_second`, `mean_edge_s`, `mean_frame_s`, `mean_track_s`, and VBG-specific `mean_integrate_s` / `mean_raycast_s`. Failed `SYCL:0` runs keep the exact error next to the CPU baseline.

Recorded 20-frame `freiburg1_xyz` baseline from the checked image:

| Mode | Device | Status | Throughput | Notes |
| --- | --- | --- | ---: | --- |
| odometry | `CPU:0` | ok | `45.7368 edges/s` | 19 tracked edges |
| dense-slam | `CPU:0` | ok | `20.1332 frames/s` | 19 tracked frames |
| vbg-slam | `CPU:0` | ok | `20.0063 frames/s` | 19 tracked frames |
| odometry | `SYCL:0` | failed |  | `UR_RESULT_ERROR_INVALID_ENUMERATION` |
| dense-slam | `SYCL:0` | failed |  | `UR_RESULT_ERROR_INVALID_ENUMERATION` |
| vbg-slam | `SYCL:0` | failed |  | `DeviceHashBackend.cpp:38: Unimplemented device` |

## Build Warning Note

During the Open3D SYCL build, Embree camera/raycasting code emits warnings such
as `Undefined function intel_has_committed_hit` and
`Undefined function intel_ray_query_abandon`. Those warnings are tied to the
Open3D/Embree SYCL raycasting path, not to basic tensor allocation. They are
consistent with `RaycastingScene` being a poor AMD ROCm target in this stack;
the core runtime blocker for tensors is still the first HIP Unified Runtime
copy/fill failure.
