# syntax=docker/dockerfile:1

# TandemLLM serving image, side-by-side with install.sh (docs/docker.md).
#
# Target: one NVIDIA DGX Spark (GB10, aarch64, sm_121). The image also builds on x86_64
# (every pin has both arches' wheels); the served profile was measured on the Spark.
#
# Base: nvcr.io/nvidia/pytorch:26.07-py3 -- we take the CUDA 13.x toolkit (nvcc) and the
# NVIDIA runtime stack from it. Its default Python is 3.12, so the 3.11 venv the engine
# needs (the Python version tests/test_grammar.py passes under) is provisioned inside by uv.
# The CUDA-availability assert deliberately lives in docker/entrypoint.sh, not here: this
# image must build without a GPU driver.
#
#   docker build --platform linux/arm64 -t tandemllm:latest .
#
# Weights are NEVER baked in. They live on a host volume mounted at /var/tandemllm/hf
# (the HF cache root; install.sh's convention ${HF_HOME:-$HOME/.cache/huggingface}). See
# docs/docker.md for the volume contract.

FROM nvcr.io/nvidia/pytorch:26.07-py3

# uv, pinned to the same version install.sh fetches when it has to.
ENV UV_VERSION=0.12.12
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
       | env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/uv/bin:${PATH}"

# Copy the engine source (the repo checkout IS the source -- the server runs from it, exactly
# like install.sh's $DIR/src). docker/.dockerignore keeps the context to what the runtime
# reads: the engine, server, tools and ops directories.
COPY . /opt/tandemllm

# Python 3.11 venv, pinned exactly to install.sh's LOCK set (docker/requirements.lock,
# copied verbatim). torch + nvidia-* come from the CUDA 13.0 index; --index-strategy
# unsafe-best-match is uv's counterpart of pip's primary+extra-index run in install.sh step 3.
# This layer needs no GPU: it only downloads wheels.
RUN uv venv --quiet --python 3.11 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
       --index-url https://pypi.org/simple \
       --extra-index-url https://download.pytorch.org/whl/cu130 \
       --index-strategy unsafe-best-match \
       -r /opt/tandemllm/docker/requirements.lock

# The entrypoint (checks, corpus build on first start, the exact ops/qwen38-engine.service
# exec line) and the host-side smoke helper.
COPY docker/entrypoint.sh /opt/tandemllm/docker/entrypoint.sh
COPY docker/smoke.sh      /opt/tandemllm/docker/smoke.sh
RUN chmod +x /opt/tandemllm/docker/entrypoint.sh /opt/tandemllm/docker/smoke.sh

# The engine's working directory: the repo checkout. All runtime paths referenced by
# ops/serve.env (NV, HEAD, CKPT8, CKPT16, CORPUS, QWEN38_MODEL, ...) are absolute and come
# from the host's serve.env / the entrypoint, not from the image.
WORKDIR /opt/tandemllm

# Defaults the entrypoint honours (override per deployment, docs/docker.md):
#   PY             the 3.11 venv's python
#   REPO           the engine source (also where ops/ lives: the serve.env's ${REPO}/ops/*.json)
#   SERVE_ENV      the serving configuration; the repo's own ops/serve.env by default
#   TANDEM_DATA_DIR the persistent data dir (corpus, kernel JIT cache, logs, usage ledger)
ENV PY=/opt/venv/bin/python \
    REPO=/opt/tandemllm \
    SERVE_ENV=/opt/tandemllm/ops/serve.env \
    TANDEM_DATA_DIR=/var/tandemllm/data \
    HF_HOME=/var/tandemllm/hf \
    TZ=Europe/Berlin

# The runtime state: hf/ is the (read-only) weights mount, data/ the persistent volume.
# Neither ever enters the image; compose mounts both under /var/tandemllm.
VOLUME ["/var/tandemllm/data"]

ENTRYPOINT ["/opt/tandemllm/docker/entrypoint.sh"]
