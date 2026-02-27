# Multi-stage build for scpn-control
# Default: python:3.12-slim. For HPC, use --build-arg BASE=rockylinux:9
ARG BASE=python:3.12-slim

# ── Builder stage ──────────────────────────────────────────────────────
FROM ${BASE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

RUN pip install --no-cache-dir maturin build && \
    cd scpn-control-rs/crates/control-python && maturin build --release && \
    cd /build && python -m build

# ── Runtime stage ──────────────────────────────────────────────────────
FROM ${BASE} AS runtime

COPY --from=builder /build/dist/*.whl /tmp/
COPY --from=builder /build/scpn-control-rs/target/wheels/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

WORKDIR /workspace
ENTRYPOINT ["scpn-control"]
CMD ["info"]
