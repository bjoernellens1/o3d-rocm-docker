# o3d-rocm-docker

Docker image for Open3D with ROCm acceleration and display support.

## Base stack

- Ubuntu 24.04
- ROCm 7.2
- Open3D built with SYCL for AMD GPUs
- Target GPU architectures: `gfx1100`, `gfx1151`

## Build locally

```bash
docker build -t o3d-rocm:local .
```

If you have a direct Codeplay oneAPI-for-AMD plugin `.deb` URL, pass it during build:

```bash
docker build \
  --build-arg CODEPLAY_AMD_PLUGIN_DEB_URL="<codeplay-deb-url>" \
  -t o3d-rocm:local .
```

## Run with display + GPU

```bash
xhost +si:localuser:$(id -un)
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  o3d-rocm:local
xhost -si:localuser:$(id -un)
```

## CI/CD

A GitHub Actions workflow builds and pushes the image to GHCR:

- `ghcr.io/<owner>/o3d-rocm:ubuntu24.04-rocm7.2`
- `ghcr.io/<owner>/o3d-rocm:latest` (default branch only)
