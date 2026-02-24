#include <stdio.h>
#include <stdint.h>

extern int rg_init();
extern int rg_cuda_available();
extern const char* rg_cuda_device_name();
extern int64_t rg_randn(int ndims, int64_t* dims, int on_cuda);
extern int rg_ndim(int64_t h);
extern int64_t rg_shape(int64_t h, int dim);
extern int64_t rg_add(int64_t a, int64_t b);
extern int64_t rg_matmul(int64_t a, int64_t b);
extern double rg_item(int64_t h);
extern int64_t rg_sum(int64_t h);
extern void rg_free_tensor(int64_t h);

int main() {
    printf("Init: %d\n", rg_init());
    printf("CUDA: %d\n", rg_cuda_available());
    printf("Device: %s\n", rg_cuda_device_name());
    
    int64_t dims[] = {3, 4};
    int64_t a = rg_randn(2, dims, 1);
    printf("Tensor a: %d dims, shape=(%ld, %ld)\n", rg_ndim(a), rg_shape(a, 0), rg_shape(a, 1));
    
    int64_t b = rg_randn(2, dims, 1);
    int64_t c = rg_add(a, b);
    printf("a + b: %d dims\n", rg_ndim(c));
    
    int64_t dims2[] = {4, 2};
    int64_t d = rg_randn(2, dims2, 1);
    int64_t e = rg_matmul(a, d);
    printf("matmul(3x4, 4x2) = (%ld, %ld)\n", rg_shape(e, 0), rg_shape(e, 1));
    
    int64_t s = rg_sum(a);
    printf("sum: %f\n", rg_item(s));
    
    rg_free_tensor(a);
    rg_free_tensor(b);
    rg_free_tensor(c);
    rg_free_tensor(d);
    rg_free_tensor(e);
    rg_free_tensor(s);
    printf("All freed. Done!\n");
    return 0;
}
