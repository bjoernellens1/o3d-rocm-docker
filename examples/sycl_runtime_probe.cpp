// Minimal DPC++/SYCL probe for the AMD HIP backend used by this image.
//
// Compile inside the container:
// icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa \
//   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1151 \
//   examples/sycl_runtime_probe.cpp -o /tmp/sycl_runtime_probe
//
// Run:
// /tmp/sycl_runtime_probe

#include <sycl/sycl.hpp>

#include <iostream>

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::cout << "queue OK: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        int *device_data = sycl::malloc_device<int>(16, q);
        std::cout << "malloc_device OK" << std::endl;

        int host_data[16] = {};
        q.copy(host_data, device_data, 16).wait();
        std::cout << "copy host->device OK" << std::endl;

        q.parallel_for(sycl::range<1>(16), [=](sycl::id<1> i) {
             device_data[i] = static_cast<int>(i[0]) + 1;
         }).wait();
        std::cout << "parallel_for OK" << std::endl;

        q.copy(device_data, host_data, 16).wait();
        std::cout << "copy device->host OK: " << host_data[0] << " "
                  << host_data[15] << std::endl;

        sycl::free(device_data, q);
    } catch (const sycl::exception &e) {
        std::cerr << "SYCL_EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
