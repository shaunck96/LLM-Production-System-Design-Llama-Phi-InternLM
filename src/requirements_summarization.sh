#!/bin/bash

# Set environment variables
export LLAMA_CUBLAS=1
export CMAKE_ARGS='-DLLAMA_CUBLAS=on'
export FORCE_CMAKE=1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

# Print environment variables to verify
echo "LLAMA_CUBLAS: $LLAMA_CUBLAS"
echo "CMAKE_ARGS: $CMAKE_ARGS"
echo "FORCE_CMAKE: $FORCE_CMAKE"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check NVIDIA GPU
nvidia-smi

# Echo CUDA-related variables
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Install llama-cpp-python with CUDA support
TORCH_CUDA_ARCH_LIST="7.0" pip install llama-cpp-python==0.2.72 --upgrade --prefer-binary --force-reinstall --no-cache-dir --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX/cu121

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install langchain_community
pip install langchain_community

echo "Installation completed!"
