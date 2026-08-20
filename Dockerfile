# Open3D 0.19 native HIP/Tensor build for the same ROCm/Python ABI used by
# ResearchFlow baseline capsules.
ARG ROCM_PYTORCH_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu22.04_py3.10_pytorch_release_2.9.1@sha256:9c9592175fece788d6c0b86059012f49b568dc95c98c13879dfdf89c30342559
FROM ${ROCM_PYTORCH_IMAGE}

ARG OPEN3D_AMD_SHA=50bb2505be991392f7cdfd040802db7bbf6ef2a8
ARG ROCM_ARCH=gfx1201
ARG AMD_MOAT_SHA=f69bb67d70e7a47af00095dab7029c230a30be73

ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_HOME=/opt/rocm \
    CUDA_HOME=/opt/rocm \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    TORCH_CUDA_ARCH_LIST="" \
    PIP_NO_CACHE_DIR=1 \
    EXPECT_BACKEND=rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ninja-build cmake pkg-config ccache \
      libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      xorg-dev libxcb-shm0 libglu1-mesa-dev libssl-dev \
      libc++-dev libc++abi-dev libsdl2-dev libxi-dev libtbb-dev \
      libosmesa6-dev libudev-dev libusb-1.0-0-dev autoconf libtool \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel 'cmake>=3.24' ninja

WORKDIR /opt
RUN git clone https://github.com/AMD-Ecosystem/Open3D.git Open3D \
    && git -C Open3D checkout --detach ${OPEN3D_AMD_SHA} \
    && test "$(git -C Open3D rev-parse HEAD)" = "${OPEN3D_AMD_SHA}" \
    && cmake -S /opt/Open3D -B /opt/Open3D/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} \
      -DPython3_EXECUTABLE=$(which python) \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_OPEN3D_ML=OFF \
      -DBUILD_ISPC_MODULE=OFF \
      -DBUILD_COMMON_CUDA_ARCHS=OFF \
    && cmake --build /opt/Open3D/build --target pip-package --parallel "$(nproc)" \
    && python -m pip install /opt/Open3D/build/lib/python_package/pip_package/open3d-*.whl \
    && rm -rf /opt/Open3D/build

COPY scripts/smoke/open3d_hip.py /opt/smoke/open3d_hip.py

LABEL org.opencontainers.image.source="https://github.com/AMD-Ecosystem/Open3D" \
      researchflow.component="open3d-rocm" \
      researchflow.backend="rocm" \
      researchflow.rocm="7.2.4" \
      researchflow.open3d.version="0.19.0" \
      researchflow.open3d.amd.sha="${OPEN3D_AMD_SHA}" \
      researchflow.amd_moat.sha="${AMD_MOAT_SHA}" \
      researchflow.gpu.arch="${ROCM_ARCH}"

WORKDIR /workspace
CMD ["python", "/opt/smoke/open3d_hip.py"]