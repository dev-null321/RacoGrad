#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

// Matrix multiplication with SIMD optimization where available
void matrix_multiply_simd(int rows_a, int cols_a, int cols_b, double *a, double *b, double *c) {
#ifdef __AVX__
    // AVX optimized version (processes 4 doubles at once)
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            __m256d sum_vec = _mm256_setzero_pd();
            
            // Process 4 elements at a time
            int k = 0;
            for (; k <= cols_a - 4; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&a[i * cols_a + k]);
                
                // For each of the 4 elements, we need to multiply by b[k+n][j]
                __m256d b_vec = _mm256_set_pd(
                    b[(k+3) * cols_b + j],
                    b[(k+2) * cols_b + j],
                    b[(k+1) * cols_b + j],
                    b[k * cols_b + j]
                );
                
                // Multiply and add
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }
            
            // Horizontal sum of the vector
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum_vec);
            double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            
            // Handle remaining elements
            for (; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            
            c[i * cols_b + j] = sum;
        }
    }
#elif defined(__SSE__)
    // SSE optimized version (processes 2 doubles at once)
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            __m128d sum_vec = _mm_setzero_pd();
            
            // Process 2 elements at a time
            int k = 0;
            for (; k <= cols_a - 2; k += 2) {
                __m128d a_vec = _mm_loadu_pd(&a[i * cols_a + k]);
                __m128d b_vec = _mm_set_pd(
                    b[(k+1) * cols_b + j],
                    b[k * cols_b + j]
                );
                
                // Multiply and add
                sum_vec = _mm_add_pd(sum_vec, _mm_mul_pd(a_vec, b_vec));
            }
            
            // Horizontal sum
            double sum_array[2];
            _mm_storeu_pd(sum_array, sum_vec);
            double sum = sum_array[0] + sum_array[1];
            
            // Handle remaining elements
            for (; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            
            c[i * cols_b + j] = sum;
        }
    }
#else
    // Fallback to standard implementation
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            c[i * cols_b + j] = sum;
        }
    }
#endif
}

// Element-wise operations with SIMD
void tensor_add_simd(int size, double *a, double *b, double *result) {
#ifdef __AVX__
    int i = 0;
    // Process 4 doubles at a time with AVX
    for (; i <= size - 4; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a[i]);
        __m256d b_vec = _mm256_loadu_pd(&b[i]);
        __m256d res_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(&result[i], res_vec);
    }
    // Handle remaining elements
    for (; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#elif defined(__SSE__)
    int i = 0;
    // Process 2 doubles at a time with SSE
    for (; i <= size - 2; i += 2) {
        __m128d a_vec = _mm_loadu_pd(&a[i]);
        __m128d b_vec = _mm_loadu_pd(&b[i]);
        __m128d res_vec = _mm_add_pd(a_vec, b_vec);
        _mm_storeu_pd(&result[i], res_vec);
    }
    // Handle remaining elements
    for (; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#else
    // Fallback implementation
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#endif
}

// Other SIMD operations (similar structure to tensor_add_simd)
void tensor_mul_elementwise_simd(int size, double *a, double *b, double *result) {
#ifdef __AVX__
    int i = 0;
    for (; i <= size - 4; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(&a[i]);
        __m256d b_vec = _mm256_loadu_pd(&b[i]);
        __m256d res_vec = _mm256_mul_pd(a_vec, b_vec);
        _mm256_storeu_pd(&result[i], res_vec);
    }
    for (; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#elif defined(__SSE__)
    int i = 0;
    for (; i <= size - 2; i += 2) {
        __m128d a_vec = _mm_loadu_pd(&a[i]);
        __m128d b_vec = _mm_loadu_pd(&b[i]);
        __m128d res_vec = _mm_mul_pd(a_vec, b_vec);
        _mm_storeu_pd(&result[i], res_vec);
    }
    for (; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#else
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#endif
}

// SIMD optimized ReLU
void relu_forward_simd(int size, double *input, double *output) {
#ifdef __AVX__
    int i = 0;
    __m256d zeros = _mm256_setzero_pd();
    
    for (; i <= size - 4; i += 4) {
        __m256d in_vec = _mm256_loadu_pd(&input[i]);
        __m256d res_vec = _mm256_max_pd(in_vec, zeros);
        _mm256_storeu_pd(&output[i], res_vec);
    }
    
    for (; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
#elif defined(__SSE__)
    int i = 0;
    __m128d zeros = _mm_setzero_pd();
    
    for (; i <= size - 2; i += 2) {
        __m128d in_vec = _mm_loadu_pd(&input[i]);
        __m128d res_vec = _mm_max_pd(in_vec, zeros);
        _mm_storeu_pd(&output[i], res_vec);
    }
    
    for (; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
#else
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
#endif
}