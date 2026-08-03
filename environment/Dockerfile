FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV VERSION=0.1.0
ENV CODE_URL=https://github.com/AllenNeuralDynamics/aind-ophys-classifier

RUN apt-get update && apt-get install -y \
    libx11-6 libxext6 libxrender1 tk \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.4.0 \
    h5py==3.12.1 \
    pint==0.24.4 \
    roicat[all]==1.4.4 \
    sparse==0.15.4

RUN mkdir -p /data /results /scratch /capsule/code
COPY code/ /capsule/code/
