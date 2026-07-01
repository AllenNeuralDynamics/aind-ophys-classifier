# hash:sha256:16d5385f4b82b5e8b7a51ad608c98bcf0cdf42919b52f38c4c1ddf43f38da5e4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION=0.1.0
ENV CODE_URL=https://github.com/allenNeuralDynamics/aind-ophys-classifier

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.4.0 \
    h5py==3.12.1 \
    pint==0.24.4 \
    roicat[all]==1.7.3 \
    sparse==0.18.0
