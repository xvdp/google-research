# https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-23-08.html  based on jax 0.4.14 it appears to be just affter optim was deprecated?

## nvcr.io bio;ds
# oldest nvcr jax is 23.08-py3 # jaxlib==0.4.14
# nvcr.io/nvidia/jax:25.08-py3 # req. cu    13.0 driver 580 
# nvcr.io/nvidia/jax:25.04-py3 # req. cu    12.9 driver 575 
# nvcr.io/nvidia/jax:25.01-py3 # req. cu    12.8 driver 570
# https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-25-01.html
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax/tags
    # JAX 0.4.38 (Includes release-specific patches and mxfp8.)
    # Flax 0.10.2
    # maxtext.git@4651cb3c73de (only in the maxtext container
    # Ubuntu 24.04 including Python 3.12
    # NVIDIA CUDA® 12.8.0.038
    # NVIDIA cuBLAS 12.8.3.14
    # NVIDIA cuDNN 9.7.0.66
    # NVIDIA NCCL 2.25.1
    # rdma-core 50.0
    # NVIDIA HPC-X 2.21
    # OpenMPI 4.1.7
    # GDRCopy 2.3.1-1
    # Nsight Compute 2025.1.0.14
    # Nsight Systems 2024.2.6.2.225
    # TransformerEngine 1.14

FROM nvcr.io/nvidia/jax:25.01-py3

# Install basic tools and system dependencies
RUN apt-get update && apt-get install -y \
    git tree ffmpeg wget build-essential cmake ninja-build \
    gcc-12 g++-12 python3 python3-venv python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
COPY ./diffusion_distillation/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt


# permit jupyter
ARG port=32778
EXPOSE $port
ENV PORT=$port
RUN pip install  --no-cache-dir jupyter
RUN echo "alias jupy='jupyter notebook --allow-root -y --no-browser --ip=0.0.0.0 --port=${PORT}'" >> ~/.bashrc
    
# build with label pointing to source file, query label
# docker build --build-arg DOCKERFILE="$(pwd)/Dockerfile" -f Dockerfile . -t google_research/xvdp/diffusion_distillation:latest
# docker inspect --format='{{index .Config.Labels "dockerfile"}}' google_research/xvdp/diffusion_distillation:latest
ARG DOCKERFILE
LABEL dockerfile="${DOCKERFILE}"
CMD ["/bin/bash"]