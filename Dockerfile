# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG RUST_TOOLCHAIN=1.88.0
ARG MATURIN_VERSION=1.11.1
ARG SOURCE_REPOSITORY=https://github.com/sycomix/Tensor-Engine.git
ARG SOURCE_REF=linux
ARG BUILD_FROM_REMOTE=0
ARG ENGINE_FEATURES=compat,opencl,openblas,safe_tensors,with_tokenizers,vision,audio,multi_precision,server
ARG PYTHON_FEATURES=python_bindings,openblas,multi_precision,safe_tensors,with_tokenizers,vision,audio,parallel_io,server,backend_wgpu,quantized,dtype_f16,dtype_bf16,compat,opencl

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    OPENBLAS_DIR=/usr \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libopenblas-dev \
        libssl-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        patchelf \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 --fail --show-error --silent https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"

ENV PATH=/root/.cargo/bin:/opt/venv/bin:${PATH}

WORKDIR /workspace
COPY . /workspace/source

RUN if [[ "${BUILD_FROM_REMOTE}" == "1" ]]; then \
        rm -rf /workspace/source; \
        git clone --depth 1 --branch "${SOURCE_REF}" "${SOURCE_REPOSITORY}" /workspace/source; \
    fi

WORKDIR /workspace/source

RUN python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "maturin==${MATURIN_VERSION}"

RUN cargo build --locked --release --bin engine --no-default-features --features "${ENGINE_FEATURES}"

RUN maturin build --locked --release --features "${PYTHON_FEATURES}" --out /opt/wheels


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=24.04
ARG SOURCE_REPOSITORY=https://github.com/sycomix/Tensor-Engine.git
ARG SOURCE_REF=linux
ARG VCS_REF=unknown
ARG IMAGE_VERSION=0.5.0-linux

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    RUST_LOG=info \
    PIP_BREAK_SYSTEM_PACKAGES=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="Tensor Engine Linux" \
      org.opencontainers.image.description="Tensor Engine Linux branch runtime with CUDA, OpenCL, OpenBLAS, Python bindings, and the engine CLI." \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.ref.name="${SOURCE_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.vendor="Sycomix"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libopenblas0-pthread \
        ocl-icd-libopencl1 \
        openssl \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/source/target/release/engine /usr/local/bin/engine
COPY --from=builder /opt/wheels /tmp/wheels

RUN python3 -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && useradd --system --create-home --home-dir /opt/tensor-engine --shell /usr/sbin/nologin tensor-engine \
    && install -d -o tensor-engine -g tensor-engine /models /data /workspace

WORKDIR /workspace
USER tensor-engine

EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/engine"]
CMD ["--help"]
