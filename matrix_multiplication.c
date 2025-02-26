#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Basic matrix multiplication
void matrix_multiply(int rows_a, int cols_a, int cols_b, double *a, double *b, double *c) {
    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols_a; ++k) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            c[i * cols_b + j] = sum;
        }
    }
}

// Add element-wise tensor operations
void tensor_add(int size, double *a, double *b, double *result) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void tensor_sub(int size, double *a, double *b, double *result) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}

void tensor_mul_elementwise(int size, double *a, double *b, double *result) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

void tensor_scale(int size, double *a, double scalar, double *result) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * scalar;
    }
}

// Activation functions in C for speed
void relu_forward(int size, double *input, double *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

void relu_backward(int size, double *input, double *output) {
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? 1 : 0;
    }
}

void sigmoid_forward(int size, double *input, double *output) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

void sigmoid_backward(int size, double *input, double *output) {
    for (int i = 0; i < size; i++) {
        double sigmoid_val = 1.0 / (1.0 + exp(-input[i]));
        output[i] = sigmoid_val * (1.0 - sigmoid_val);
    }
}