# hash:sha256:13d3244d31affc371c58631bf831c03b2eef9fd4cb671c7e38defc38bef441a4
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
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-classifier-library.git@e2d83e3#egg=aind-ophys-classifier-library
