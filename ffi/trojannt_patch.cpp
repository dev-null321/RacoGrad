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
    return ;
}

int64_t rg_loaded_param_handle(int idx) {
    if (idx >= 0 && idx < (int)loaded_model_params.size())
        return loaded_model_params[idx].second;
    return -1;
}
