# TUM Open3D CPU vs ROCm Benchmark

- Sequence: `/data/tum_rgbd/freiburg1_xyz/rgbd_dataset_freiburg1_xyz`
- Max frames: `20`
- Generated: `2026-05-21T09:40:52+0000`

| Mode | Device | Status | FPS / edges/s | Mean frame s | Mean track s | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| odometry | `CPU:0` | ok | 45.7368 | 0.0209 |  | tracked=19 |
| odometry | `SYCL:0` | failed |  |  |  | `RuntimeError: hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| dense-slam | `CPU:0` | ok | 20.1332 | 0.0497 | 0.0123 | tracked=19 |
| dense-slam | `SYCL:0` | failed |  |  |  | `RuntimeError: hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| vbg-slam | `CPU:0` | ok | 20.0063 | 0.0499 | 0.0083 | tracked=19 |
| vbg-slam | `SYCL:0` | failed |  |  |  | `RuntimeError: [Open3D Error] (std::shared_ptr<DeviceHashBackend> open3d::core::CreateDeviceHashBackend(int64_t, const Dtype &, const SizeVector &, const std::vector<Dtype> &, const std::vector<SizeVector> &, const Device &, const HashBackendType &)) /tmp/Open3D/cpp/open3d/core/hashmap/DeviceHashBackend.cpp:38: Unimplemented device` |

