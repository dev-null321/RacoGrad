#!/bin/bash
# Compile RacoGrad llama.cpp FFI bridge
# Produces ffi/libracograd_llama_ffi.so

LLAMA_DIR="/home/marq/Projects/multivac/llama.cpp"

echo "Compiling RacoGrad llama.cpp FFI..."
g++ -shared -fPIC -o ffi/libracograd_llama_ffi.so ffi/racograd_llama_ffi.cpp \
  -I${LLAMA_DIR}/include \
  -I${LLAMA_DIR}/ggml/include \
  -L${LLAMA_DIR}/build/bin \
  -lllama -lggml -lggml-base \
  -Wl,-rpath,${LLAMA_DIR}/build/bin \
  -std=c++17 -O2

if [ $? -eq 0 ]; then
    echo "Done. Output: ffi/libracograd_llama_ffi.so"
    ls -lh ffi/libracograd_llama_ffi.so
else
    echo "Compilation failed!"
    exit 1
fi
