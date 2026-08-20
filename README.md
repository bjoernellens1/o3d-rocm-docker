# o3d-rocm-docker

Qualified Open3D Tensor images for AMD GPUs using AMD's native HIP port.

## Canonical stack

- ROCm **7.2.4**
- Ubuntu 22.04 / Python 3.10 ABI matching the ResearchFlow baseline capsules
- AMD ROCm PyTorch 2.9.1 base pinned by OCI digest
- `AMD-Ecosystem/Open3D` pinned to `50bb2505be991392f7cdfd040802db7bbf6ef2a8`
- Open3D 0.19 native `USE_HIP=ON` backend
- per-architecture images for `gfx1201`, `gfx1151`, and `gfx90a`

This replaces the earlier SYCL/Codeplay experiment. AMD's Open3D port reuses the normal Open3D CUDA/Tensor implementation while compiling the GPU translation units through HIP. The Python-facing device remains `o3d.core.Device("CUDA:0")`, which allows existing Open3D Tensor users such as VarSplat and SGAD-SLAM to keep their device-facing code unchanged.

## Build

```bash
docker build \
  --build-arg ROCM_ARCH=gfx1201 \
  -t o3d-rocm:local .
```

AMD Open3D currently requires CMake >=3.24; the image installs a pinned-compatible modern CMake before configuring the source tree.

## Run the qualification smoke

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  o3d-rocm:local
```

The smoke is intentionally stronger than an import test. It must prove on a real AMD GPU that all of the following work:

- `o3d.core.cuda.is_available()`
- `o3d.core.Device("CUDA:0")`
- GPU Tensor allocation and arithmetic
- `o3d.t.geometry.Image` / `RGBDImage`
- `o3d.t.pipelines.odometry.rgbd_odometry_multi_scale`

Those capabilities are the subset required by the retained RGB-D SLAM baselines.

## CI/CD and image identity

The workflow builds candidates for `gfx1201`, `gfx1151`, and `gfx90a`, publishes immutable source-SHA tags, and then executes the Open3D smoke on matching self-hosted AMD runners. Only a candidate that passes the real GPU smoke is promoted to a convenience tag such as:

```text
ghcr.io/bjoernellens1/o3d-rocm:rocm-7.2.4-gfx1201
```

Scientific consumers should use the resulting immutable `image@sha256:<digest>` recorded in the qualification artifact rather than a moving tag.

The scheduled weekly workflow only runs when this workflow exists on the repository's default branch. Until this portability branch is made the default branch or merged, push-triggered qualification works but the weekly schedule does not.
