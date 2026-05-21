FROM docker.io/rocm/dev-ubuntu-24.04:7.2

ARG DEBIAN_FRONTEND=noninteractive
ARG OPEN3D_VERSION=v0.19.0
ARG ROCM_ARCHS="gfx1100;gfx1151"
ARG SYCL_BACKEND=source
ARG UNIFIED_RUNTIME_REF=08ebfcbab17e1f606d12f9bb2963e6c1bcbfa161
ARG CODEPLAY_AMD_PLUGIN_DEB_URL=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/open3d-venv \
    DISPLAY=:0 \
    QT_X11_NO_MITSHM=1 \
    ONEAPI_DEVICE_SELECTOR=hip:* \
    O3D_SYCL_BACKEND=${SYCL_BACKEND}

ENV PATH="${VIRTUAL_ENV}/bin:/opt/intel/oneapi/compiler/2026.0/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8:/opt/intel/oneapi/tcm/1.5/lib:/opt/intel/oneapi/umf/1.1/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}"
ENV MKLROOT=/opt/intel/oneapi/mkl/latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    software-properties-common \
    build-essential \
    ninja-build \
    ccache \
    cmake \
    pkg-config \
    libhwloc-dev \
    libc++-dev \
    libc++abi-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 \
    libglu1-mesa-dev \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxkbcommon0 \
    libxkbcommon-dev \
    libwayland-dev \
    wayland-protocols \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp \
      intel-oneapi-mkl-devel \
      intel-oneapi-tbb-devel && \
    ln -s libmkl_sycl.so /opt/intel/oneapi/mkl/latest/lib/libmkl_sycl.a && \
    rm -rf /var/lib/apt/lists/*

ENV MKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl
ENV TBB_DIR=/opt/intel/oneapi/tbb/2023.0/lib/cmake/tbb
ENV CPATH="/opt/intel/oneapi/dpl/latest/include:${CPATH}" \
    LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib:${LIBRARY_PATH}"

COPY patches/unified-runtime-hip-device-info.patch /tmp/unified-runtime-hip-device-info.patch

RUN set -eux; \
    case "${SYCL_BACKEND}" in \
      source|codeplay|none) ;; \
      *) echo "SYCL_BACKEND must be one of: source, codeplay, none" >&2; exit 2 ;; \
    esac; \
    if [ "${SYCL_BACKEND}" = "codeplay" ]; then \
      if [ -z "${CODEPLAY_AMD_PLUGIN_DEB_URL}" ]; then \
        echo "SYCL_BACKEND=codeplay requires CODEPLAY_AMD_PLUGIN_DEB_URL" >&2; \
        exit 2; \
      fi; \
      curl -fsSL "${CODEPLAY_AMD_PLUGIN_DEB_URL}" -o /tmp/codeplay-amd-plugin.deb; \
      apt-get update; \
      apt-get install -y --no-install-recommends /tmp/codeplay-amd-plugin.deb; \
      rm -f /tmp/codeplay-amd-plugin.deb; \
      rm -rf /var/lib/apt/lists/*; \
    elif [ -n "${CODEPLAY_AMD_PLUGIN_DEB_URL}" ]; then \
      echo "Ignoring CODEPLAY_AMD_PLUGIN_DEB_URL because SYCL_BACKEND=${SYCL_BACKEND}" >&2; \
    fi

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel

RUN set -eux; \
    if [ "${SYCL_BACKEND}" = "source" ]; then \
      git init /tmp/unified-runtime; \
      git -C /tmp/unified-runtime remote add origin https://github.com/oneapi-src/unified-runtime.git; \
      git -C /tmp/unified-runtime fetch --depth 1 origin "${UNIFIED_RUNTIME_REF}"; \
      git -C /tmp/unified-runtime checkout --detach FETCH_HEAD; \
      git -C /tmp/unified-runtime apply /tmp/unified-runtime-hip-device-info.patch; \
      cmake -S /tmp/unified-runtime -B /tmp/unified-runtime/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUR_BUILD_ADAPTER_HIP=ON \
        -DUR_BUILD_ADAPTER_CUDA=OFF \
        -DUR_BUILD_ADAPTER_L0=OFF \
        -DUR_BUILD_ADAPTER_OPENCL=OFF \
        -DUR_BUILD_ADAPTER_NATIVE_CPU=OFF \
        -DUR_BUILD_TESTS=OFF \
        -DUR_BUILD_EXAMPLES=OFF; \
      cmake --build /tmp/unified-runtime/build --target ur_adapter_hip --parallel "$(nproc)"; \
      cp -a /tmp/unified-runtime/build/lib/libur_adapter_hip.so* /opt/intel/oneapi/compiler/2026.0/lib/; \
      rm -rf /tmp/unified-runtime; \
      ldconfig; \
    else \
      echo "Skipping source-built Unified Runtime HIP adapter for SYCL_BACKEND=${SYCL_BACKEND}"; \
    fi

RUN set -eux; \
    git clone --depth 1 --branch "${OPEN3D_VERSION}" https://github.com/isl-org/Open3D.git /tmp/Open3D; \
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'sed -i \' \
      '  -e "s|return this_sub_group().shuffle(x, local_id);|return sycl::select_from_group(this_sub_group(), x, local_id);|" \' \
      '  -e "s|return this_sub_group().shuffle_down(x, delta);|return sycl::shift_group_left(this_sub_group(), x, delta);|" \' \
      '  -e "s|return this_sub_group().shuffle_up(x, delta);|return sycl::shift_group_right(this_sub_group(), x, delta);|" \' \
      '  "$1"' \
      > /tmp/Open3D/patch-embree-sycl.sh; \
    chmod +x /tmp/Open3D/patch-embree-sycl.sh; \
    perl -0pi -e 's#(ExternalProject_Add\(\s*ext_embree\s*)#$1\n    PATCH_COMMAND /tmp/Open3D/patch-embree-sycl.sh <SOURCE_DIR>/common/sys/sycl.h\n#s' /tmp/Open3D/3rdparty/embree/embree.cmake; \
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'sed -i -e "s|A.rows ;|A.rows() ;|g" "$1"' \
      > /tmp/Open3D/patch-poissonrecon.sh; \
    chmod +x /tmp/Open3D/patch-poissonrecon.sh; \
    perl -0pi -e 's#(ExternalProject_Add\(\s*ext_poisson\s*)#$1\n    PATCH_COMMAND /tmp/Open3D/patch-poissonrecon.sh <SOURCE_DIR>/Src/SparseMatrix.inl\n#s' /tmp/Open3D/3rdparty/possionrecon/possionrecon.cmake; \
    sed -i \
      -e 's/set(MKL_LINK static)/set(MKL_LINK dynamic)/' \
      -e 's|LIB_DIR      ${MKL_ROOT}/lib/intel64|LIB_DIR      ${MKL_ROOT}/lib|' \
      /tmp/Open3D/3rdparty/find_dependencies.cmake; \
    open3d_build_sycl=ON; \
    if [ "${SYCL_BACKEND}" = "none" ]; then \
      open3d_build_sycl=OFF; \
    fi; \
    sycl_arch_flags=""; \
    for arch in $(echo "${ROCM_ARCHS}" | tr ';' ' '); do \
      sycl_arch_flags="${sycl_arch_flags}-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${arch} "; \
    done; \
    cmake -S /tmp/Open3D -B /tmp/Open3D/build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_SYCL_MODULE="${open3d_build_sycl}" \
      -DBUILD_GUI=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_PREFIX_PATH="${MKLROOT};/opt/intel/oneapi/tbb/2023.0" \
      -DCMAKE_EXE_LINKER_FLAGS="-L${MKLROOT}/lib" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L${MKLROOT}/lib" \
      -DCMAKE_MODULE_LINKER_FLAGS="-L${MKLROOT}/lib" \
      -DMKL_DIR="${MKL_DIR}" \
      -DMKL_LINK=dynamic \
      -DMKL_SYCL_LINK=dynamic \
      -DTBB_DIR="${TBB_DIR}" \
      -DPython3_EXECUTABLE="${VIRTUAL_ENV}/bin/python" \
      -DCMAKE_SYCL_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa ${sycl_arch_flags}" \
      -DSYCL_TARGETS="${ROCM_ARCHS}"; \
    cmake --build /tmp/Open3D/build --target pip-package --parallel "$(nproc)"; \
    python -m pip install /tmp/Open3D/build/lib/python_package/pip_package/open3d-*.whl; \
    rm -rf /tmp/Open3D

ENV LD_PRELOAD="/opt/intel/oneapi/mkl/latest/lib/libmkl_intel_lp64.so:/opt/intel/oneapi/mkl/latest/lib/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/libmkl_tbb_thread.so"

WORKDIR /workspace

CMD ["python", "-c", "import open3d as o3d; print('Open3D', o3d.__version__); print('SYCL devices:', o3d.core.sycl.get_available_devices())"]
