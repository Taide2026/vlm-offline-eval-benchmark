# aarch64 / GB10 (DGX Spark). CUDA devel base matching the host toolkit
# (13.0, V13.0.88): vLLM's aarch64 wheel links libcudart.so.13, and
# flashinfer JIT-compiles kernels at runtime with nvcc — it needs the full
# coherent toolkit (nvcc + curand/cccl headers), same as /usr/local/cuda-13.0
# provides on the host. The pip-bundled nvidia/cu13 toolkit mixes nvcc 13.2
# with 13.0 headers and lacks curand.h, so it can't build flashinfer.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# gcc/g++: host compiler for nvcc and triton's runtime JIT.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy

# Dependencies first so code edits don't invalidate the 15+ GB layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra vllm

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra vllm

ENV HF_HOME=/root/.cache/huggingface
ENV CUDA_HOME=/usr/local/cuda
# .venv/bin on PATH: JIT subprocesses need ninja (and python) from the venv,
# which `uv run` would normally provide.
ENV PATH=/app/.venv/bin:$CUDA_HOME/bin:$PATH
ENTRYPOINT ["/app/.venv/bin/realtime-bench"]
