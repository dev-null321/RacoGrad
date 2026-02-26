// racograd_ffi.cpp — C bridge between Racket FFI and libtorch
// Zero Python dependency. Racket → C → libtorch → CUDA
//
// Compile:
//   g++ -shared -fPIC -o libracograd_ffi.so racograd_ffi.cpp \
//     -I/home/marq/libtorch-install/libtorch/include \
//     -I/home/marq/libtorch-install/libtorch/include/torch/csrc/api/include \
//     -L/home/marq/libtorch-install/libtorch/lib \
//     -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
//     -Wl,-rpath,/home/marq/libtorch-install/libtorch/lib \
//     -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

// ============================================================
// Handle system: opaque pointers to torch::Tensor and Optimizer
// ============================================================

std::unordered_map<int64_t, torch::Tensor> tensor_store;
static std::unordered_map<int64_t, std::shared_ptr<torch::optim::Adam>> adam_store;
int64_t next_handle = 1;

int64_t store_tensor(torch::Tensor t) {
    int64_t h = next_handle++;
    tensor_store[h] = std::move(t);
    return h;
}

torch::Tensor& get_tensor(int64_t h) {
    return tensor_store.at(h);
}

extern "C" {

// ============================================================
// Lifecycle
// ============================================================

int rg_init() {
    // Nothing to init — libtorch initializes on first use
    return torch::cuda::is_available() ? 1 : 0;
}

int rg_cuda_available() {
    return torch::cuda::is_available() ? 1 : 0;
}

const char* rg_cuda_device_name() {
    static std::string name;
    if (torch::cuda::is_available()) {
        // Get device properties
        auto props = at::cuda::getDeviceProperties((c10::DeviceIndex)0);
        name = props->name;
    } else {
        name = "none";
    }
    return name.c_str();
}

void rg_free_tensor(int64_t h) {
    tensor_store.erase(h);
}

void rg_sync() {
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }
}

// ============================================================
// Tensor Creation
// ============================================================

int64_t rg_zeros(int ndims, int64_t* dims, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (on_cuda && torch::cuda::is_available())
        options = options.device(torch::kCUDA);
    std::vector<int64_t> shape(dims, dims + ndims);
    return store_tensor(torch::zeros(shape, options));
}

int64_t rg_ones(int ndims, int64_t* dims, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (on_cuda && torch::cuda::is_available())
        options = options.device(torch::kCUDA);
    std::vector<int64_t> shape(dims, dims + ndims);
    return store_tensor(torch::ones(shape, options));
}

int64_t rg_randn(int ndims, int64_t* dims, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (on_cuda && torch::cuda::is_available())
        options = options.device(torch::kCUDA);
    std::vector<int64_t> shape(dims, dims + ndims);
    return store_tensor(torch::randn(shape, options));
}

int64_t rg_full(int ndims, int64_t* dims, double val, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    if (on_cuda && torch::cuda::is_available())
        options = options.device(torch::kCUDA);
    std::vector<int64_t> shape(dims, dims + ndims);
    return store_tensor(torch::full(shape, val, options));
}

int64_t rg_arange(int64_t start, int64_t end, int64_t step, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kLong);
    if (on_cuda && torch::cuda::is_available())
        options = options.device(torch::kCUDA);
    return store_tensor(torch::arange(start, end, step, options));
}

int64_t rg_tensor_from_float(double val) {
    auto t = torch::tensor(val, torch::TensorOptions().dtype(torch::kFloat32));
    if (torch::cuda::is_available()) t = t.to(torch::kCUDA);
    return store_tensor(t);
}

// Create tensor from flat data + shape
int64_t rg_tensor_from_data(int ndims, int64_t* dims, double* data, int nelems, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // Create on CPU first, then move
    auto t = torch::from_blob(data, {nelems}, options).clone();
    std::vector<int64_t> shape(dims, dims + ndims);
    t = t.reshape(shape);
    if (on_cuda && torch::cuda::is_available()) t = t.to(torch::kCUDA);
    return store_tensor(t);
}

int64_t rg_tensor_from_long_data(int ndims, int64_t* dims, int64_t* data, int nelems, int on_cuda) {
    auto options = torch::TensorOptions().dtype(torch::kLong);
    auto t = torch::from_blob(data, {nelems}, options).clone();
    std::vector<int64_t> shape(dims, dims + ndims);
    t = t.reshape(shape);
    if (on_cuda && torch::cuda::is_available()) t = t.to(torch::kCUDA);
    return store_tensor(t);
}

// ============================================================
// Tensor Info
// ============================================================

int rg_ndim(int64_t h) {
    return get_tensor(h).dim();
}

int64_t rg_shape(int64_t h, int dim) {
    return get_tensor(h).size(dim);
}

double rg_item(int64_t h) {
    return get_tensor(h).item<double>();
}

int64_t rg_numel(int64_t h) {
    return get_tensor(h).numel();
}

// Copy data to CPU float buffer
void rg_to_float_buffer(int64_t h, double* out, int n) {
    auto t = get_tensor(h).cpu().to(torch::kFloat64).contiguous();
    memcpy(out, t.data_ptr<double>(), n * sizeof(double));
}

// ============================================================
// Device / Type
// ============================================================

int64_t rg_to_cuda(int64_t h) {
    return store_tensor(get_tensor(h).to(torch::kCUDA));
}

int64_t rg_to_cpu(int64_t h) {
    return store_tensor(get_tensor(h).cpu());
}

int64_t rg_to_float(int64_t h) {
    return store_tensor(get_tensor(h).to(torch::kFloat32));
}

int64_t rg_to_long(int64_t h) {
    return store_tensor(get_tensor(h).to(torch::kLong));
}

int64_t rg_to_int(int64_t h) {
    return store_tensor(get_tensor(h).to(torch::kInt32));
}

int rg_is_cuda(int64_t h) {
    return get_tensor(h).is_cuda() ? 1 : 0;
}

// ============================================================
// Requires Grad
// ============================================================

int64_t rg_set_requires_grad(int64_t h, int val) {
    get_tensor(h).requires_grad_(val != 0);
    return h;
}

int rg_requires_grad(int64_t h) {
    return get_tensor(h).requires_grad() ? 1 : 0;
}

// ============================================================
// Basic Math (element-wise, return new tensor)
// ============================================================

int64_t rg_add(int64_t a, int64_t b) {
    return store_tensor(get_tensor(a) + get_tensor(b));
}

int64_t rg_sub(int64_t a, int64_t b) {
    return store_tensor(get_tensor(a) - get_tensor(b));
}

int64_t rg_mul(int64_t a, int64_t b) {
    return store_tensor(get_tensor(a) * get_tensor(b));
}

int64_t rg_div(int64_t a, int64_t b) {
    return store_tensor(get_tensor(a) / get_tensor(b));
}

int64_t rg_neg(int64_t h) {
    return store_tensor(-get_tensor(h));
}

int64_t rg_abs(int64_t h) {
    return store_tensor(torch::abs(get_tensor(h)));
}

int64_t rg_sqrt(int64_t h) {
    return store_tensor(torch::sqrt(get_tensor(h)));
}

int64_t rg_exp(int64_t h) {
    return store_tensor(torch::exp(get_tensor(h)));
}

int64_t rg_log(int64_t h) {
    return store_tensor(torch::log(get_tensor(h)));
}

int64_t rg_sin(int64_t h) {
    return store_tensor(torch::sin(get_tensor(h)));
}

int64_t rg_cos(int64_t h) {
    return store_tensor(torch::cos(get_tensor(h)));
}

int64_t rg_tanh(int64_t h) {
    return store_tensor(torch::tanh(get_tensor(h)));
}

int64_t rg_sigmoid(int64_t h) {
    return store_tensor(torch::sigmoid(get_tensor(h)));
}

int64_t rg_pow(int64_t h, double exp) {
    return store_tensor(torch::pow(get_tensor(h), exp));
}

// ============================================================
// Reduction
// ============================================================

int64_t rg_sum(int64_t h) {
    return store_tensor(get_tensor(h).sum());
}

int64_t rg_sum_dim(int64_t h, int dim, int keepdim) {
    return store_tensor(get_tensor(h).sum(dim, keepdim != 0));
}

int64_t rg_mean(int64_t h) {
    return store_tensor(get_tensor(h).mean());
}

int64_t rg_mean_dim(int64_t h, int dim, int keepdim) {
    return store_tensor(get_tensor(h).mean(dim, keepdim != 0));
}

int64_t rg_max(int64_t h) {
    return store_tensor(std::get<0>(get_tensor(h).max(0)));
}

int64_t rg_min(int64_t h) {
    return store_tensor(std::get<0>(get_tensor(h).min(0)));
}

int64_t rg_argmax(int64_t h, int dim) {
    return store_tensor(get_tensor(h).argmax(dim));
}

// ============================================================
// Matrix ops
// ============================================================

int64_t rg_matmul(int64_t a, int64_t b) {
    return store_tensor(torch::matmul(get_tensor(a), get_tensor(b)));
}

int64_t rg_mm(int64_t a, int64_t b) {
    return store_tensor(torch::mm(get_tensor(a), get_tensor(b)));
}

int64_t rg_bmm(int64_t a, int64_t b) {
    return store_tensor(torch::bmm(get_tensor(a), get_tensor(b)));
}

int64_t rg_transpose(int64_t h, int dim0, int dim1) {
    return store_tensor(get_tensor(h).transpose(dim0, dim1));
}

// ============================================================
// Shape ops
// ============================================================

int64_t rg_reshape(int64_t h, int ndims, int64_t* dims) {
    std::vector<int64_t> shape(dims, dims + ndims);
    return store_tensor(get_tensor(h).reshape(shape));
}

int64_t rg_unsqueeze(int64_t h, int dim) {
    return store_tensor(get_tensor(h).unsqueeze(dim));
}

int64_t rg_squeeze(int64_t h, int dim) {
    return store_tensor(get_tensor(h).squeeze(dim));
}

int64_t rg_cat(int64_t* handles, int n, int dim) {
    std::vector<torch::Tensor> tensors;
    for (int i = 0; i < n; i++)
        tensors.push_back(get_tensor(handles[i]));
    return store_tensor(torch::cat(tensors, dim));
}

int64_t rg_stack(int64_t* handles, int n, int dim) {
    std::vector<torch::Tensor> tensors;
    for (int i = 0; i < n; i++)
        tensors.push_back(get_tensor(handles[i]));
    return store_tensor(torch::stack(tensors, dim));
}

int64_t rg_slice(int64_t h, int dim, int64_t start, int64_t end) {
    return store_tensor(get_tensor(h).slice(dim, start, end));
}

int64_t rg_contiguous(int64_t h) {
    return store_tensor(get_tensor(h).contiguous());
}

// ============================================================
// Mask ops
// ============================================================

int64_t rg_triu(int64_t h, int diagonal) {
    return store_tensor(torch::triu(get_tensor(h), diagonal));
}

int64_t rg_tril(int64_t h, int diagonal) {
    return store_tensor(torch::tril(get_tensor(h), diagonal));
}

int64_t rg_masked_fill(int64_t h, int64_t mask, double val) {
    return store_tensor(get_tensor(h).masked_fill(get_tensor(mask), val));
}

int64_t rg_eq(int64_t a, int64_t b) {
    return store_tensor(get_tensor(a).eq(get_tensor(b)));
}

// ============================================================
// NN ops
// ============================================================

int64_t rg_relu(int64_t h) {
    return store_tensor(torch::relu(get_tensor(h)));
}

int64_t rg_gelu(int64_t h) {
    return store_tensor(torch::gelu(get_tensor(h)));
}

int64_t rg_softmax(int64_t h, int dim) {
    return store_tensor(torch::softmax(get_tensor(h), dim));
}

int64_t rg_dropout(int64_t h, double p, int training) {
    return store_tensor(torch::dropout(get_tensor(h), p, training != 0));
}

int64_t rg_layer_norm(int64_t h, int64_t gamma, int64_t beta, int norm_size, double eps) {
    std::vector<int64_t> shape = {norm_size};
    return store_tensor(torch::layer_norm(get_tensor(h), shape,
                                           get_tensor(gamma), get_tensor(beta), eps));
}

// Layer norm without gamma/beta (just normalize)
int64_t rg_layer_norm_simple(int64_t h, int norm_size, double eps) {
    std::vector<int64_t> shape = {norm_size};
    return store_tensor(torch::layer_norm(get_tensor(h), shape, {}, {}, eps));
}

int64_t rg_embedding(int64_t weight, int64_t indices) {
    return store_tensor(torch::embedding(get_tensor(weight), get_tensor(indices)));
}

int64_t rg_linear(int64_t input, int64_t weight, int64_t bias) {
    if (bias == 0) {
        return store_tensor(torch::linear(get_tensor(input), get_tensor(weight)));
    }
    return store_tensor(torch::linear(get_tensor(input), get_tensor(weight), get_tensor(bias)));
}

// ============================================================
// Loss functions
// ============================================================

int64_t rg_cross_entropy_loss(int64_t logits, int64_t targets, int64_t ignore_index) {
    auto l = get_tensor(logits);
    auto t = get_tensor(targets);
    // Auto-flatten 3D logits: (batch, seq, vocab) -> (batch*seq, vocab)
    if (l.dim() == 3) {
        auto batch = l.size(0);
        auto seq = l.size(1);
        auto vocab = l.size(2);
        l = l.reshape({batch * seq, vocab});
        t = t.reshape({batch * seq});
    }
    return store_tensor(torch::nn::functional::cross_entropy(
        l, t,
        torch::nn::functional::CrossEntropyFuncOptions().ignore_index(ignore_index)));
}

int64_t rg_mse_loss(int64_t input, int64_t target) {
    return store_tensor(torch::mse_loss(get_tensor(input), get_tensor(target)));
}

int64_t rg_nll_loss(int64_t input, int64_t target) {
    return store_tensor(torch::nll_loss(get_tensor(input), get_tensor(target)));
}

// ============================================================
// Autograd
// ============================================================

void rg_backward(int64_t h) {
    get_tensor(h).backward();
}

int64_t rg_grad(int64_t h) {
    return store_tensor(get_tensor(h).grad().clone());
}

void rg_zero_grad(int64_t h) {
    if (get_tensor(h).grad().defined()) {
        get_tensor(h).grad().zero_();
    }
}

int64_t rg_detach(int64_t h) {
    return store_tensor(get_tensor(h).detach());
}

int64_t rg_no_grad_begin() {
    // Use torch::NoGradGuard in a scoped manner
    // For C API, we'll use the inference mode
    torch::NoGradGuard* guard = new torch::NoGradGuard();
    return (int64_t)guard;
}

void rg_no_grad_end(int64_t guard_handle) {
    delete (torch::NoGradGuard*)guard_handle;
}

// ============================================================
// Optimizer: Adam
// ============================================================

int64_t rg_adam_create(int64_t* param_handles, int n_params, 
                        double lr, double beta1, double beta2, 
                        double weight_decay) {
    std::vector<torch::Tensor> params;
    for (int i = 0; i < n_params; i++) {
        params.push_back(get_tensor(param_handles[i]));
    }
    
    auto options = torch::optim::AdamOptions(lr)
        .betas({beta1, beta2})
        .weight_decay(weight_decay);
    
    auto optimizer = std::make_shared<torch::optim::Adam>(params, options);
    int64_t h = next_handle++;
    adam_store[h] = optimizer;
    return h;
}

void rg_adam_step(int64_t h) {
    adam_store.at(h)->step();
}

void rg_adam_zero_grad(int64_t h) {
    adam_store.at(h)->zero_grad();
}

void rg_adam_free(int64_t h) {
    adam_store.erase(h);
}

// ============================================================
// Misc
// ============================================================

int64_t rg_clone(int64_t h) {
    return store_tensor(get_tensor(h).clone());
}

int64_t rg_copy(int64_t dst, int64_t src) {
    get_tensor(dst).copy_(get_tensor(src));
    return dst;
}

void rg_print(int64_t h) {
    std::cout << get_tensor(h) << std::endl;
}

// Einsum
int64_t rg_einsum(const char* equation, int64_t* handles, int n) {
    std::vector<torch::Tensor> tensors;
    for (int i = 0; i < n; i++)
        tensors.push_back(get_tensor(handles[i]));
    return store_tensor(torch::einsum(equation, tensors));
}

// Clamp
int64_t rg_clamp(int64_t h, double min_val, double max_val) {
    return store_tensor(torch::clamp(get_tensor(h), min_val, max_val));
}

// TrojaNNt FFI patch — fixes for libtorch 2.5.1 API

// SVD singular values only
int64_t rg_svdvals(int64_t h) {
    try {
        auto sv = torch::linalg::svdvals(get_tensor(h).to(torch::kFloat64), c10::nullopt);
        return store_tensor(sv);
    } catch (...) {
        return -1;
    }
}

// Full SVD
int rg_svd_full(int64_t h, int64_t* u_out, int64_t* s_out, int64_t* v_out) {
    try {
        auto result = torch::linalg::svd(get_tensor(h).to(torch::kFloat64), false, c10::nullopt);
        *u_out = store_tensor(std::get<0>(result));
        *s_out = store_tensor(std::get<1>(result));
        *v_out = store_tensor(std::get<2>(result));
        return 0;
    } catch (...) {
        return -1;
    }
}

// Statistics
double rg_std_val(int64_t h) {
    return get_tensor(h).to(torch::kFloat64).std().item<double>();
}
double rg_var_val(int64_t h) {
    return get_tensor(h).to(torch::kFloat64).var().item<double>();
}
double rg_min_val(int64_t h) {
    return get_tensor(h).to(torch::kFloat64).min().item<double>();
}
double rg_max_val(int64_t h) {
    return get_tensor(h).to(torch::kFloat64).max().item<double>();
}
double rg_mean_val(int64_t h) {
    return get_tensor(h).to(torch::kFloat64).mean().item<double>();
}
double rg_frobenius_norm(int64_t h) {
    return torch::norm(get_tensor(h).to(torch::kFloat64)).item<double>();
}

// L2 norm along dimension
int64_t rg_norm_dim(int64_t h, int dim) {
    return store_tensor(torch::norm(get_tensor(h).to(torch::kFloat64), 2, {dim}));
}

// Counting
int64_t rg_count_above(int64_t h, double threshold) {
    return (get_tensor(h).abs() > threshold).sum().item<int64_t>();
}
int64_t rg_count_near_zero(int64_t h, double eps) {
    return (get_tensor(h).abs() < eps).sum().item<int64_t>();
}

// Reshape, flatten, sort, abs
int64_t rg_reshape_2d(int64_t h, int64_t rows, int64_t cols) {
    try {
        return store_tensor(get_tensor(h).reshape({rows, cols}).contiguous().to(torch::kFloat64));
    } catch (...) { return -1; }
}
int64_t rg_flatten_tensor(int64_t h) {
    return store_tensor(get_tensor(h).flatten());
}
int64_t rg_sort_desc(int64_t h) {
    auto result = get_tensor(h).sort(0, true);
    return store_tensor(std::get<0>(result));
}
int64_t rg_abs_tensor(int64_t h) {
    return store_tensor(get_tensor(h).abs());
}
double rg_get_double(int64_t h, int64_t idx) {
    return get_tensor(h).to(torch::kFloat64).flatten()[idx].item<double>();
}

// Dimension info
int rg_tensor_dim(int64_t h) {
    return (int)get_tensor(h).dim();
}
int64_t rg_size_at(int64_t h, int dim) {
    return get_tensor(h).size(dim);
}

// Load TorchScript model parameters
static std::vector<std::pair<std::string, int64_t>> loaded_model_params;

int rg_load_model_params(const char* path) {
    try {
        loaded_model_params.clear();
        auto module = torch::jit::load(path, torch::kCPU);
        for (const auto& param : module.named_parameters()) {
            int64_t h = store_tensor(param.value.detach().clone());
            loaded_model_params.push_back({param.name, h});
        }
        return (int)loaded_model_params.size();
    } catch (...) { return -1; }
}

const char* rg_loaded_param_name(int idx) {
    static thread_local std::string buf;
    if (idx >= 0 && idx < (int)loaded_model_params.size()) {
        buf = loaded_model_params[idx].first;
        return buf.c_str();
    }
    return "";
}

int64_t rg_loaded_param_handle(int idx) {
    if (idx >= 0 && idx < (int)loaded_model_params.size())
        return loaded_model_params[idx].second;
    return -1;
}

} // extern "C"
