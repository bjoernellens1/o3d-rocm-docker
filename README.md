# o3d-rocm-docker

Container and repro examples for evaluating Open3D tensor workloads on AMD ROCm.
The practical baseline here is ROCm 7.2 on `gfx1151`; Open3D SYCL on AMD is
treated as an experimental probe, not a reliable acceleration path.

## Base Stack

- Ubuntu 24.04
- ROCm 7.2
- Intel oneAPI DPC++ compiler/runtime
- Open3D v0.19.0 built from source
- Target GPU architectures: `gfx1100`, `gfx1151`

## Build Locally

The default build uses a pinned source build of the Unified Runtime HIP adapter
and applies a small device-info patch so `sycl-ls --verbose` can enumerate the
AMD HIP device cleanly.

```bash
podman build \
  --build-arg SYCL_BACKEND=source \
  -t localhost/o3d-rocm:local .
```

Supported backend modes:

- `SYCL_BACKEND=source`: build Open3D with SYCL and install a pinned source-built HIP adapter.
- `SYCL_BACKEND=none`: build Open3D without the SYCL module for CPU-only checks.
- `SYCL_BACKEND=codeplay`: install a Codeplay AMD plugin `.deb` from `CODEPLAY_AMD_PLUGIN_DEB_URL`.

Codeplay mode is intentionally explicit because the public Codeplay AMD plugin
matrix does not currently line up cleanly with ROCm 7.2 / `gfx1151`.

```bash
podman build \
  --build-arg SYCL_BACKEND=codeplay \
  --build-arg CODEPLAY_AMD_PLUGIN_DEB_URL="<codeplay-deb-url>" \
  -t localhost/o3d-rocm:codeplay .
```

## Run On Fedora Podman

For rootless Fedora Podman with AMD GPU passthrough, use `/dev/kfd`,
`/dev/dri`, group passthrough, disabled SELinux relabeling for the workspace
mount, unconfined seccomp, and host IPC:

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
  localhost/o3d-rocm:local
```

## Current Result

The built image sees `SYCL:0`, but useful Open3D tensor execution on the ROCm
SYCL path is not working on this stack. Basic allocation succeeds, while the
first host-to-device tensor copy/fill fails with:

```text
hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)
```

CPU Open3D tensor odometry, dense SLAM, and VoxelBlockGrid SLAM examples work
and emit timing metrics. See [examples/README.md](/home/bjoern/git/o3d-rocm-docker/examples/README.md)
for exact commands and recorded outputs.

## CI/CD

A GitHub Actions workflow builds and pushes the image to GHCR:

- `ghcr.io/<owner>/o3d-rocm:ubuntu24.04-rocm7.2`
- `ghcr.io/<owner>/o3d-rocm:latest` on the default branch
