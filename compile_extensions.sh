#!/bin/bash

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SHARED_EXT=".dylib"
    COMPILE_FLAGS="-dynamiclib -O3"
    if [[ $(sysctl -n machdep.cpu.features) == *"AVX"* ]]; then
        SIMD_FLAGS="-mavx"
    elif [[ $(sysctl -n machdep.cpu.features) == *"SSE"* ]]; then
        SIMD_FLAGS="-msse4.2"
    else
        SIMD_FLAGS=""
    fi
else
    # Linux and others
    SHARED_EXT=".so"
    COMPILE_FLAGS="-shared -fPIC -O3"
    if grep -q avx /proc/cpuinfo; then
        SIMD_FLAGS="-mavx"
    elif grep -q sse /proc/cpuinfo; then
        SIMD_FLAGS="-msse4.2"
    else
        SIMD_FLAGS=""
    fi
fi

# Compile basic matrix operations
echo "Compiling basic matrix operations..."
cc $COMPILE_FLAGS -o matrix_multiplication$SHARED_EXT matrix_multiplication.c

# Compile SIMD operations
echo "Compiling SIMD optimized operations..."
cc $COMPILE_FLAGS $SIMD_FLAGS -o simd_ops$SHARED_EXT simd_ops.c

# Compile parallel operations
echo "Compiling parallel operations..."
cc $COMPILE_FLAGS -o parallel_ops$SHARED_EXT parallel_ops.c -lpthread

# Compile OpenCL operations
if [ "$(uname)" == "Darwin" ]; then
    echo "Compiling OpenCL operations (macOS)..."
    cc $COMPILE_FLAGS -o matrix_opencl$SHARED_EXT matrix_opencl.c -framework OpenCL
else
    echo "Compiling OpenCL operations (Linux)..."
    cc $COMPILE_FLAGS -o matrix_opencl$SHARED_EXT matrix_opencl.c -lOpenCL
fi

# Compile MLX placeholder operations (for Apple Silicon)
echo "Compiling MLX placeholder operations..."
cc $COMPILE_FLAGS -o mlx_ops$SHARED_EXT mlx_ops.c

# Compile CUDA placeholder operations
echo "Compiling CUDA placeholder operations..."
cc $COMPILE_FLAGS -o cuda_ops$SHARED_EXT cuda_ops.c

# Compile CNN operations
echo "Compiling CNN operations..."
cc $COMPILE_FLAGS -o cnn_ops$SHARED_EXT cnn_ops.c

# Compile MLX CNN operations (for Apple Silicon)
echo "Compiling MLX CNN operations..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    cc $COMPILE_FLAGS -o mlx_cnn_ops$SHARED_EXT mlx_cnn_ops.c
else
    cc $COMPILE_FLAGS -fopenmp -o mlx_cnn_ops$SHARED_EXT mlx_cnn_ops.c
fi

echo "Compilation complete!"
echo
echo "=================================================================="
echo "OPTIMIZATION SUMMARY"
echo "=================================================================="
echo "1. C Extensions: Basic operations in C for improved performance"
echo "2. SIMD Vectorization: Using CPU vector instructions when available"
echo "   SIMD flags used: $SIMD_FLAGS"
echo "3. Parallel Processing: Multi-threaded operations for batch processing"
echo "4. Memory Optimization: Better memory layout and in-place operations"
echo "5. GPU Acceleration: See gpu_acceleration.md for implementation options"
echo "=================================================================="