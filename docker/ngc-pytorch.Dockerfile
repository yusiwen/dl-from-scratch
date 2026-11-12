# NVIDIA NGC PyTorch image for the GB10 GPU (Grace Blackwell, compute capability
# 12.1), plus the one repository dependency the official image does not ship.
#
# Build:  docker build -t dlf-ngc:26.09 -f docker/ngc-pytorch.Dockerfile docker/
# Run:    scripts/ngc-run.sh python -m cv.simplecnn.train
FROM nvcr.io/nvidia/pytorch:26.09-py3

# ml/basics/metrics.py imports scikit-learn. The image already provides numpy,
# datasets, pyyaml, tensorboard, torchvision and matplotlib.
RUN python -m pip install --no-cache-dir "scikit-learn==1.9.0"
