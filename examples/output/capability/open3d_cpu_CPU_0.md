# Open3D SYCL Capability Probe

- Device: `CPU:0`
- Generated: `2026-05-21T09:53:13+0000`
- Open3D: `0.19.0+1e7b174`
- SYCL devices: `['SYCL:0']`
- O3D_SYCL_BACKEND: `source`

| Stage | Case | Expected SYCL status | Result | Elapsed s | Error |
| --- | --- | --- | --- | ---: | --- |
| 00_device | sycl_discovery | must pass before any SYCL workload can run | ok | 0.0756 |  |
| 01_tensor_basic | tensor_empty_allocation | core tensor allocation | ok | 0.0002 |  |
| 01_tensor_basic | tensor_host_copy_reduction | host-to-device copy plus reduction | ok | 0.0002 |  |
| 01_tensor_basic | tensor_zeros_reduction | fill kernel plus reduction | ok | 0.0002 |  |
| 02_tensor_kernels | tensor_elementwise_reduction | elementwise kernels plus reduction | ok | 0.0003 |  |
| 02_tensor_kernels | tensor_linalg_matmul_inv | SYCL linalg path if MKL SYCL backend is working | ok | 0.0073 |  |
| 03_tensor_geometry | image_to_device | tensor image copy | ok | 0.0001 |  |
| 03_tensor_geometry | raycasting_scene | Open3D has a SYCL RaycastingScene implementation | ok | 0.0015 |  |
| 04_registration | tensor_icp | ICP pose kernels dispatch CPU/CUDA only after basic tensor ops work | ok | 0.0071 |  |
| 05_rgbd | rgbd_odometry | RGB-D odometry kernels dispatch CPU/CUDA only in Open3D v0.19 | ok | 0.0054 |  |
| 06_vbg_slam | voxel_block_grid | expected CPU/CUDA-only hash backend in Open3D v0.19 source | ok | 0.0001 |  |
| 06_vbg_slam | dense_slam_model | inherits VoxelBlockGrid and RGB-D odometry limitations | ok | 0.0002 |  |
