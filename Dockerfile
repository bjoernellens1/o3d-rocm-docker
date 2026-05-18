FROM rocm/dev-ubuntu-24.04:7.2

ARG DEBIAN_FRONTEND=noninteractive
ARG OPEN3D_VERSION=v0.19.0
ARG OPEN3D_WHEEL_VERSION=0.19.0
ARG ROCM_ARCHS="gfx1100;gfx1151"
ARG CODEPLAY_AMD_PLUGIN_DEB_URL=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:0 \
    QT_X11_NO_MITSHM=1 \
    SYCL_DEVICE_FILTER=amd

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
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 \
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
    libxkbcommon0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.repos.intel.com/oneapi/intel-oneapi-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp \
      intel-oneapi-tbb-devel && \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ -n "${CODEPLAY_AMD_PLUGIN_DEB_URL}" ]; then \
      curl -fsSL "${CODEPLAY_AMD_PLUGIN_DEB_URL}" -o /tmp/codeplay-amd-plugin.deb; \
      apt-get update; \
      apt-get install -y --no-install-recommends /tmp/codeplay-amd-plugin.deb; \
      rm -f /tmp/codeplay-amd-plugin.deb; \
      rm -rf /var/lib/apt/lists/*; \
    fi

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN set -eux; \
    mkdir -p /tmp/open3d-wheel; \
    if python3 -m pip download --only-binary=:all: --no-deps "open3d==${OPEN3D_WHEEL_VERSION}" -d /tmp/open3d-wheel; then \
      python3 -m pip install /tmp/open3d-wheel/open3d-*.whl; \
      if python3 -c "import open3d as o3d, sys; ok = hasattr(o3d.core, 'sycl') and hasattr(o3d.core.sycl, 'get_available_devices'); print('SYCL APIs missing from prebuilt wheel' if not ok else 'SYCL APIs available in prebuilt wheel'); sys.exit(0 if ok else 1)"; then \
        rm -rf /tmp/open3d-wheel; \
        exit 0; \
      fi; \
      echo "Prebuilt Open3D wheel does not expose SYCL APIs; falling back to source build."; \
      python3 -m pip uninstall -y open3d; \
    fi; \
    rm -rf /tmp/open3d-wheel; \
    git clone --depth 1 --branch "${OPEN3D_VERSION}" https://github.com/isl-org/Open3D.git /tmp/Open3D; \
    . /opt/intel/oneapi/setvars.sh; \
    sycl_arch_flags=""; \
    for arch in $(echo "${ROCM_ARCHS}" | tr ';' ' '); do \
      sycl_arch_flags="${sycl_arch_flags}-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${arch} "; \
    done; \
    cmake -S /tmp/Open3D -B /tmp/Open3D/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_SYCL_MODULE=ON \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DPython3_EXECUTABLE=/usr/bin/python3 \
      -DCMAKE_SYCL_FLAGS="-fsycl -fsycl-targets=amdgcn-amd-amdhsa ${sycl_arch_flags}" \
      -DSYCL_TARGETS="${ROCM_ARCHS}"; \
    cmake --build /tmp/Open3D/build --target pip-package --parallel "$(nproc)"; \
    python3 -m pip install /tmp/Open3D/build/lib/python_package/pip_package/open3d-*.whl; \
    rm -rf /tmp/Open3D

WORKDIR /workspace

CMD ["python3", "-c", "import open3d as o3d; print('Open3D', o3d.__version__); print('SYCL devices:', o3d.core.sycl.get_available_devices())"]
