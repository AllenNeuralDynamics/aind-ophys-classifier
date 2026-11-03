# hash:sha256:bf7eebb5f0bcde5adc959daa1a12c647094a4d73509316e2255b27b1167bffeb
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
    -e git+https://github.com/AllenNeuralDynamics/aind-ophys-classifier-library.git@v0.2.1#egg=aind-ophys-classifier-library
