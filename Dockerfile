# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.

# ── Builder stage ──────────────────────────────────────────────────────
FROM python:3.12-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Pin rustup installer to known version
ARG RUSTUP_VERSION=1.28.1
RUN curl --proto '=https' --tlsv1.2 -sSf \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
        -o /tmp/rustup-init && \
    chmod +x /tmp/rustup-init && \
    /tmp/rustup-init -y --default-toolchain stable && \
    rm /tmp/rustup-init

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

RUN pip install --require-hashes -r requirements/ci-build.txt -r requirements/ci-interop.txt && \
    cd scpn-control-rs/crates/control-python && maturin build --release && \
    cd /build && python -m build

# ── Runtime stage ──────────────────────────────────────────────────────
FROM python:3.12-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS runtime

COPY --from=builder /build/dist/*.whl /tmp/
COPY --from=builder /build/scpn-control-rs/target/wheels/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

WORKDIR /workspace
ENTRYPOINT ["scpn-control"]
CMD ["info"]
