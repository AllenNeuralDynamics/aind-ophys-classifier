# hash:sha256:930406afa1fd79d717bb13db73edff2d89884a390e2c7d305cf5102b1ce727c1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION=0.1.0
ENV CODE_URL=https://github.com/allenNeuralDynamics/aind-ophys-classifier

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-classifier-library.git@8367e2f#egg=aind-ophys-classifier-library
