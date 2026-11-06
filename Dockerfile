# syntax=docker/dockerfile:1
#
# Runpod Pod template: interactive RL training in MuJoCo physics simulation.
#   JupyterLab (8888) + SSH (22) come from the official Runpod PyTorch base's /start.sh.
#   TensorBoard (6006) is started by /post_start.sh, a hook that /start.sh runs.
#   Stack: Gymnasium + MuJoCo + Stable-Baselines3 (+ sb3-contrib) + a CleanRL-style PPO.
#
# Build (from this directory, on any x86_64 Docker builder or the GitHub workflow):
#   docker build --platform linux/amd64 -t <registry>/<user>/sim-rl-train:0.1.0 .

# Official Runpod PyTorch 2.8.0 image (template runpod-torch-v280):
# Ubuntu 24.04, Python 3.12, CUDA 12.8.1, torch 2.8.0+cu128.
# Pin the digest for fully reproducible builds:  ...ubuntu2404@sha256:<digest>
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# MUJOCO_GL=egl: headless GPU rendering for env.render()/videos (no X server).
#   On a host where EGL fails, set MUJOCO_GL=osmesa (CPU software rendering).
# NVIDIA_DRIVER_CAPABILITIES=all: the NVIDIA runtime then mounts libEGL_nvidia.
# PIP_BREAK_SYSTEM_PACKAGES: Ubuntu 24.04 marks system Python as externally
#   managed; torch lives there, so extra packages go into the same interpreter.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    RL_RUNS_DIR=/workspace/runs \
    SIMRL_HOME=/opt/simrl

# OpenGL/EGL/OSMesa runtime libraries for MuJoCo rendering + ffmpeg for videos.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglfw3 ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 # Register NVIDIA's EGL vendor library with glvnd (often missing in containers,
 # which makes EGL silently fall back to Mesa or fail).
 && mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Heavy Python layer: changes only when requirements.txt changes.
# The constraints file pins the base image's torch so pip cannot swap it out.
COPY requirements.txt /opt/simrl/requirements.txt
RUN python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' > /opt/simrl/constraints.txt \
 && cat /opt/simrl/constraints.txt \
 && python -m pip install -r /opt/simrl/requirements.txt -c /opt/simrl/constraints.txt \
 && python -c "import torch, mujoco, gymnasium, stable_baselines3, sb3_contrib; \
print('torch', torch.__version__, '| mujoco', mujoco.__version__, '| gymnasium', gymnasium.__version__, '| sb3', stable_baselines3.__version__)"

# Light layer: scripts and notebook (rebuilt on every code change).
COPY simrl/ /opt/simrl/
COPY post_start.sh /post_start.sh
RUN chmod +x /post_start.sh /opt/simrl/*.py \
 && python -m py_compile /opt/simrl/*.py

# Documentation only; the Runpod template must expose these ports.
EXPOSE 22 6006 8888

# No CMD/ENTRYPOINT override: the base image's /start.sh starts nginx, SSH
# (if PUBLIC_KEY is set), JupyterLab (if JUPYTER_PASSWORD is set), runs
# /post_start.sh, then stays alive with `sleep infinity`.
