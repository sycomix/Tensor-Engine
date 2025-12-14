# Dockerfile for reproducible builds of llava training environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl build-essential pkg-config libssl-dev ca-certificates libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Rust + maturin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"
RUN python3 -m pip install --upgrade pip && python3 -m pip install maturin

WORKDIR /work
COPY examples/llava_project/requirements.txt requirements.txt
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install -r requirements.txt

# Build Tensor-Engine from the provided repository (local repo must be part of the build context)
WORKDIR /work
COPY . /work
RUN . /work/venv/bin/activate && pip install maturin && python3 -m maturin develop --release --features "python_bindings,vision,with_tokenizers,openblas"

# Run the as_any_mut verification as a sanity check to catch automation regressions
RUN python scripts/verify_as_any_mut.py || (echo "as_any_mut verification failed" && exit 1)

# Default command: run a quick synthetic smoke training (real training requires a dataset manifest).
ENTRYPOINT ["bash", "-lc", ". venv/bin/activate && python -m examples.train_llava --synthetic --epochs 1 --batch 2"]
