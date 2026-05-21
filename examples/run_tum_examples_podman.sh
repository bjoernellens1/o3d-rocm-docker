#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-localhost/o3d-rocm:local}"
TUM_ROOT="${TUM_ROOT:-/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd}"
SEQUENCE="${SEQUENCE:-freiburg1_xyz/rgbd_dataset_freiburg1_xyz}"
DEVICE="${DEVICE:-CPU:0}"
MAX_FRAMES="${MAX_FRAMES:-30}"
MODE="${MODE:-odometry}"
DEVICE_SAFE="${DEVICE//:/_}"
CAPABILITY_KIND="sycl"
if [[ "${DEVICE}" == CPU:* ]]; then
  CAPABILITY_KIND="cpu"
fi
PODMAN_TTY_ARGS=(-i)
if [ -t 1 ]; then
  PODMAN_TTY_ARGS+=(-t)
fi

podman run --rm "${PODMAN_TTY_ARGS[@]}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add keep-groups \
  --security-opt=label=disable \
  --security-opt=seccomp=unconfined \
  --ipc=host \
  --userns=keep-id \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$(pwd):/workspace:rw" \
  -v "${TUM_ROOT}:/data/tum_rgbd:ro" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    case '${MODE}' in
      sycl-smoke)
        python examples/open3d_rocm_sycl_smoke.py --device '${DEVICE}' --json
        ;;
      capability|sycl-capability)
        python examples/open3d_sycl_capability_probe.py \
          --device '${DEVICE}' \
          --json-out 'examples/output/capability/open3d_${CAPABILITY_KIND}_${DEVICE_SAFE}.json' \
          --md-out 'examples/output/capability/open3d_${CAPABILITY_KIND}_${DEVICE_SAFE}.md'
        ;;
      odometry)
        python examples/tum_tensor_rgbd_odometry.py \
          --sequence '/data/tum_rgbd/${SEQUENCE}' \
          --device '${DEVICE}' \
          --max-frames '${MAX_FRAMES}' \
          --json-out 'examples/output/tum_odometry_${DEVICE_SAFE}.json'
        ;;
      slam)
        python examples/tum_dense_slam.py \
          --sequence '/data/tum_rgbd/${SEQUENCE}' \
          --device '${DEVICE}' \
          --max-frames '${MAX_FRAMES}' \
          --json-out 'examples/output/tum_dense_slam_${DEVICE_SAFE}.json' \
          --pointcloud-out 'examples/output/tum_dense_slam_${DEVICE_SAFE}.ply'
        ;;
      vbg-slam)
        python examples/tum_vbg_slam.py \
          --sequence '/data/tum_rgbd/${SEQUENCE}' \
          --device '${DEVICE}' \
          --max-frames '${MAX_FRAMES}' \
          --json-out 'examples/output/tum_vbg_slam_${DEVICE_SAFE}.json' \
          --pointcloud-out 'examples/output/tum_vbg_slam_${DEVICE_SAFE}.ply'
        ;;
      benchmark)
        python examples/benchmark_tum_cpu_vs_rocm.py \
          --sequence '/data/tum_rgbd/${SEQUENCE}' \
          --max-frames '${MAX_FRAMES}' \
          --output-dir 'examples/output/benchmark'
        ;;
      *)
        echo 'MODE must be one of: sycl-smoke, capability, odometry, slam, vbg-slam, benchmark' >&2
        exit 2
        ;;
    esac
  "
