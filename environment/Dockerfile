# hash:sha256:99cfc43eaa0029f35154505ace69aff54a00c422fea544f72f7c44c35f3ba837
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV VERSION=0.1.0
ENV CODE_URL=https://github.com/allenNeuralDynamics/aind-ophys-classifier

RUN pip install -U --no-cache-dir \
    aind-data-schema \
    h5py==3.12.1 \
    pint==0.24.4 \
    roicat[all]==1.4.4 \
    sparse==0.15.4
