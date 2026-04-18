// racograd_llama_ffi.cpp — C bridge between Racket FFI and llama.cpp
// Zero Python dependency. Racket → C → llama.cpp → CUDA
//
// Compile:
//   g++ -shared -fPIC -o ffi/libracograd_llama_ffi.so ffi/racograd_llama_ffi.cpp \
//     -I/home/marq/Projects/multivac/llama.cpp/include \
//     -I/home/marq/Projects/multivac/llama.cpp/ggml/include \
//     -L/home/marq/Projects/multivac/llama.cpp/build/bin \
//     -lllama -lggml -lggml-base \
//     -Wl,-rpath,/home/marq/Projects/multivac/llama.cpp/build/bin \
//     -std=c++17 -O2

#include "llama.h"
#include "ggml-backend.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdio>

// ============================================================
// Handle system: opaque int64 handles for Racket FFI
// Mirrors the pattern from racograd_ffi.cpp
// ============================================================

static std::unordered_map<int64_t, llama_model*>   model_store;
static std::unordered_map<int64_t, llama_context*>  ctx_store;
static std::unordered_map<int64_t, llama_sampler*>  sampler_store;
static int64_t next_llama_handle = 1;

extern "C" {

// ============================================================
// Backend lifecycle
// ============================================================

void rg_llama_backend_init() {
    ggml_backend_load_all();
    llama_backend_init();
}

void rg_llama_backend_free() {
    llama_backend_free();
}

// ============================================================
// Model
// ============================================================

int64_t rg_llama_model_load(const char* path, int32_t n_gpu_layers) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;

    llama_model* model = llama_model_load_from_file(path, params);
    if (!model) {
        fprintf(stderr, "rg_llama: failed to load model: %s\n", path);
        return -1;
    }

    int64_t h = next_llama_handle++;
    model_store[h] = model;
    return h;
}

void rg_llama_model_free(int64_t h) {
    auto it = model_store.find(h);
    if (it != model_store.end()) {
        llama_model_free(it->second);
        model_store.erase(it);
    }
}

int32_t rg_llama_n_vocab(int64_t model_h) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_vocab_n_tokens(vocab);
}

int32_t rg_llama_model_desc(int64_t model_h, char* buf, int32_t buf_len) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    return llama_model_desc(it->second, buf, (size_t)buf_len);
}

int32_t rg_llama_model_n_layer(int64_t model_h) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    return llama_model_n_layer(it->second);
}

int32_t rg_llama_model_n_embd(int64_t model_h) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    return llama_model_n_embd(it->second);
}

// ============================================================
// Context
// ============================================================

// Cache-type ints are GGML_TYPE_* values from ggml.h (F16=1, Q4_0=2, Q8_0=8).
// The Racket backend is the only caller and is responsible for validation;
// this layer just forwards the int to params.type_k / params.type_v.
int64_t rg_llama_ctx_create(int64_t model_h,
                            uint32_t n_ctx,
                            uint32_t n_batch,
                            int32_t cache_type_k_int,
                            int32_t cache_type_v_int) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;

    llama_context_params params = llama_context_default_params();
    params.n_ctx   = n_ctx;
    params.n_batch = n_batch;
    params.type_k  = (ggml_type) cache_type_k_int;
    params.type_v  = (ggml_type) cache_type_v_int;
    params.no_perf = false;

    llama_context* ctx = llama_init_from_model(it->second, params);
    if (!ctx) {
        fprintf(stderr, "rg_llama: failed to create context\n");
        return -1;
    }

    int64_t h = next_llama_handle++;
    ctx_store[h] = ctx;
    return h;
}

void rg_llama_ctx_free(int64_t h) {
    auto it = ctx_store.find(h);
    if (it != ctx_store.end()) {
        llama_free(it->second);
        ctx_store.erase(it);
    }
}

// ============================================================
// Tokenization
// ============================================================

int32_t rg_llama_tokenize(int64_t model_h, const char* text, int32_t text_len,
                          int32_t* out_tokens, int32_t max_tokens, int32_t add_bos) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;

    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_tokenize(vocab, text, text_len, (llama_token*)out_tokens,
                          max_tokens, add_bos != 0, true);
}

int32_t rg_llama_detokenize(int64_t model_h, int32_t token, char* buf, int32_t buf_len) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;

    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_token_to_piece(vocab, (llama_token)token, buf, buf_len, 0, true);
}

int32_t rg_llama_bos_token(int64_t model_h) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_vocab_bos(vocab);
}

int32_t rg_llama_eos_token(int64_t model_h) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_vocab_eos(vocab);
}

int32_t rg_llama_is_eog(int64_t model_h, int32_t token) {
    auto it = model_store.find(model_h);
    if (it == model_store.end()) return 0;
    const llama_vocab* vocab = llama_model_get_vocab(it->second);
    return llama_vocab_is_eog(vocab, (llama_token)token) ? 1 : 0;
}

// ============================================================
// Decode
// ============================================================

int32_t rg_llama_decode(int64_t ctx_h, int32_t* tokens, int32_t n_tokens) {
    auto it = ctx_store.find(ctx_h);
    if (it == ctx_store.end()) return -1;
    // This matches llama.cpp's examples/simple/simple.cpp — `llama_decode`
    // auto-assigns positions for batches built via `llama_batch_get_one`.
    llama_batch batch = llama_batch_get_one((llama_token*)tokens, n_tokens);
    return llama_decode(it->second, batch);
}

// ============================================================
// Logits
// ============================================================

float* rg_llama_get_logits(int64_t ctx_h) {
    auto it = ctx_store.find(ctx_h);
    if (it == ctx_store.end()) return nullptr;
    return llama_get_logits(it->second);
}

float* rg_llama_get_logits_ith(int64_t ctx_h, int32_t i) {
    auto it = ctx_store.find(ctx_h);
    if (it == ctx_store.end()) return nullptr;
    return llama_get_logits_ith(it->second, i);
}

// ============================================================
// Sampler
// ============================================================

int64_t rg_llama_sampler_create(uint32_t seed, int32_t top_k, float top_p, float temp) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    int64_t h = next_llama_handle++;
    sampler_store[h] = smpl;
    return h;
}

int64_t rg_llama_sampler_create_greedy() {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int64_t h = next_llama_handle++;
    sampler_store[h] = smpl;
    return h;
}

int32_t rg_llama_sampler_sample(int64_t sampler_h, int64_t ctx_h, int32_t idx) {
    auto sit = sampler_store.find(sampler_h);
    auto cit = ctx_store.find(ctx_h);
    if (sit == sampler_store.end() || cit == ctx_store.end()) return -1;

    return llama_sampler_sample(sit->second, cit->second, idx);
}

void rg_llama_sampler_free(int64_t h) {
    auto it = sampler_store.find(h);
    if (it != sampler_store.end()) {
        llama_sampler_free(it->second);
        sampler_store.erase(it);
    }
}

// ============================================================
// Perf
// ============================================================

void rg_llama_perf_context_print(int64_t ctx_h) {
    auto it = ctx_store.find(ctx_h);
    if (it != ctx_store.end()) {
        llama_perf_context_print(it->second);
    }
}

void rg_llama_perf_sampler_print(int64_t sampler_h) {
    auto it = sampler_store.find(sampler_h);
    if (it != sampler_store.end()) {
        llama_perf_sampler_print(it->second);
    }
}

} // extern "C"
