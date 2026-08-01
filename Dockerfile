# GPU training base — match CUDA to your cluster (12.4 is a common choice)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # Headless MuJoCo on servers (no display)
    MUJOCO_GL=egl

# System packages: Python + build tools + MuJoCo/OpenGL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy source needed for editable install
COPY pyproject.toml README.md ./
COPY teleopit/ teleopit/
COPY train_mimic/ train_mimic/
COPY scripts/ scripts/

# Install Teleopit + training extras (rsl-rl-lib, mjlab, wandb, etc.)
RUN python -m pip install --no-cache-dir -U pip setuptools wheel && \
    python -m pip install --no-cache-dir -e ".[train]"


WORKDIR /app