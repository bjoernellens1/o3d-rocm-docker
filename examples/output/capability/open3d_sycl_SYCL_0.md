# Open3D SYCL Capability Probe

- Device: `SYCL:0`
- Generated: `2026-05-21T09:53:22+0000`
- Open3D: `0.19.0+1e7b174`
- SYCL devices: `['SYCL:0']`
- O3D_SYCL_BACKEND: `source`

| Stage | Case | Expected SYCL status | Result | Elapsed s | Error |
| --- | --- | --- | --- | ---: | --- |
| 00_device | sycl_discovery | must pass before any SYCL workload can run | ok | 0.0861 |  |
| 01_tensor_basic | tensor_empty_allocation | core tensor allocation | ok | 0.0922 |  |
| 01_tensor_basic | tensor_host_copy_reduction | host-to-device copy plus reduction | failed | 0.0816 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 01_tensor_basic | tensor_zeros_reduction | fill kernel plus reduction | failed | 0.0823 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 02_tensor_kernels | tensor_elementwise_reduction | elementwise kernels plus reduction | failed | 0.0801 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 02_tensor_kernels | tensor_linalg_matmul_inv | SYCL linalg path if MKL SYCL backend is working | failed | 0.0852 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 03_tensor_geometry | image_to_device | tensor image copy | failed | 0.0818 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 03_tensor_geometry | raycasting_scene | Open3D has a SYCL RaycastingScene implementation | failed | 0.0755 | `[Open3D Error] (void open3d::t::geometry::RaycastingScene::SYCLImpl::InitializeDevice()) /tmp/Open3D/cpp/open3d/t/geometry/RaycastingScene.cpp:459: Caught exception creating sycl::device: No device of requested type available. Please check https://software.intel.com/content/www/us/en/develop/article` |
| 04_registration | tensor_icp | ICP pose kernels dispatch CPU/CUDA only after basic tensor ops work | failed | 0.0756 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 05_rgbd | rgbd_odometry | RGB-D odometry kernels dispatch CPU/CUDA only in Open3D v0.19 | failed | 0.0781 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
| 06_vbg_slam | voxel_block_grid | expected CPU/CUDA-only hash backend in Open3D v0.19 source | failed | 0.0005 | `[Open3D Error] (std::shared_ptr<DeviceHashBackend> open3d::core::CreateDeviceHashBackend(int64_t, const Dtype &, const SizeVector &, const std::vector<Dtype> &, const std::vector<SizeVector> &, const Device &, const HashBackendType &)) /tmp/Open3D/cpp/open3d/core/hashmap/DeviceHashBackend.cpp:38: Un` |
| 06_vbg_slam | dense_slam_model | inherits VoxelBlockGrid and RGB-D odometry limitations | failed | 0.0808 | `hip backend failed with error: 53 (UR_RESULT_ERROR_INVALID_ENUMERATION)` |
