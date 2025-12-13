# Dockerfile for reproducible builds of llava training environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl build-essential pkg-config libssl-dev ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Rust + maturin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"
RUN python3 -m pip install --upgrade pip && python3 -m pip install maturin

WORKDIR /work
COPY requirements.txt requirements.txt
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install -r requirements.txt

# Build Tensor-Engine from the provided repository (local repo must be part of the build context)
WORKDIR /work
COPY . /work
RUN . /work/venv/bin/activate && pip install maturin && python3 -m maturin develop --release --features "python_bindings,vision"

# Default command: run a quick training to validate
ENTRYPOINT ["bash", "-lc", ". venv/bin/activate && python -m examples.train_llava --epochs 1 --batch 2"]
