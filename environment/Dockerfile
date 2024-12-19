# hash:sha256:56e254a27b859ecf6faf475f1efd0b6df9d5a994674640c234e0a7232e754310
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.12.1 \
    pint==0.24.4 \
    roicat[all]==1.4.4 \
    sparse==0.15.4
