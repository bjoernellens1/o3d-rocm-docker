# Canonical Open3D Tensor-on-AMD candidate for ResearchFlow baselines.
# AMD-Ecosystem/Open3D provides native HIP support through USE_HIP=ON.
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
FROM ${ROCM_BASE} AS builder

ARG OPEN3D_AMD_SHA=50bb2505be991392f7cdfd040802db7bbf6ef2a8
ARG AMD_MOAT_SHA=f69bb67d70e7a47af00095dab7029c230a30be73
ARG ROCM_ARCH=gfx1201
ARG MINIFORGE_VERSION=24.7.1-2
ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_HOME=/opt/rocm \
    CUDA_HOME=/opt/rocm \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git wget build-essential pkg-config ccache ninja-build \
      xorg-dev libxcb-shm0 libglu1-mesa-dev libssl-dev libc++-dev libc++abi-dev \
      libsdl2-dev libxi-dev libtbb-dev libosmesa6-dev libudev-dev libusb-1.0-0-dev \
      autoconf libtool libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:${PATH}
RUN conda create -y -n open3d python=3.10 pip \
    && conda clean -afy
ENV PATH=/opt/conda/envs/open3d/bin:/opt/conda/bin:${PATH}
RUN python -m pip install --upgrade pip setuptools wheel 'cmake>=3.24'

WORKDIR /opt
RUN git clone https://github.com/AMD-Ecosystem/Open3D.git Open3D \
    && git -C Open3D checkout --detach ${OPEN3D_AMD_SHA} \
    && test "$(git -C Open3D rev-parse HEAD)" = "${OPEN3D_AMD_SHA}"

RUN cmake -S /opt/Open3D -B /opt/Open3D/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_HIP=ON \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} \
      -DPython3_EXECUTABLE=/opt/conda/envs/open3d/bin/python \
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
    && mkdir -p /wheelhouse \
    && cp /opt/Open3D/build/lib/python_package/pip_package/open3d-*.whl /wheelhouse/

FROM ${ROCM_BASE} AS runtime
ARG OPEN3D_AMD_SHA=50bb2505be991392f7cdfd040802db7bbf6ef2a8
ARG AMD_MOAT_SHA=f69bb67d70e7a47af00095dab7029c230a30be73
ARG ROCM_ARCH=gfx1201
ARG MINIFORGE_VERSION=24.7.1-2
ENV ROCM_HOME=/opt/rocm CUDA_HOME=/opt/rocm PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:${PATH}
RUN conda create -y -n open3d python=3.10 pip numpy \
    && conda clean -afy
ENV PATH=/opt/conda/envs/open3d/bin:/opt/conda/bin:${PATH}
COPY --from=builder /wheelhouse /wheelhouse
RUN python -m pip install /wheelhouse/open3d-*.whl
COPY scripts/smoke/open3d_tensor_rocm.py /opt/smoke/open3d_tensor_rocm.py

LABEL org.opencontainers.image.source="https://github.com/AMD-Ecosystem/Open3D" \
      researchflow.component="open3d-rocm" \
      researchflow.open3d.amd.sha="${OPEN3D_AMD_SHA}" \
      researchflow.amd_moat.sha="${AMD_MOAT_SHA}" \
      researchflow.backend="rocm" \
      researchflow.rocm="7.2.4" \
      researchflow.rocm_base.digest="sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054" \
      researchflow.gpu_arch="${ROCM_ARCH}" \
      researchflow.capability.open3d_tensor_gpu="candidate" \
      researchflow.capability.open3d_rgbd_odometry_gpu="candidate"

CMD ["python", "/opt/smoke/open3d_tensor_rocm.py"]
