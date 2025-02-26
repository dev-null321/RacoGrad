#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Thread arguments for parallel matrix multiplication
typedef struct {
    int start_row;
    int end_row;
    int cols_a;
    int cols_b;
    double *a;
    double *b;
    double *c;
} thread_data_t;

// Thread function for matrix multiplication
void* matrix_multiply_thread(void *arg) {
    thread_data_t *data = (thread_data_t*)arg;
    
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->cols_b; j++) {
            double sum = 0.0;
            for (int k = 0; k < data->cols_a; k++) {
                sum += data->a[i * data->cols_a + k] * data->b[k * data->cols_b + j];
            }
            data->c[i * data->cols_b + j] = sum;
        }
    }
    
    pthread_exit(NULL);
}

// Parallel matrix multiplication
void matrix_multiply_parallel(int rows_a, int cols_a, int cols_b, double *a, double *b, double *c, int num_threads) {
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];
    
    int rows_per_thread = rows_a / num_threads;
    int remainder = rows_a % num_threads;
    
    int start_row = 0;
    for (int t = 0; t < num_threads; t++) {
        int end_row = start_row + rows_per_thread + (t < remainder ? 1 : 0);
        
        thread_data[t].start_row = start_row;
        thread_data[t].end_row = end_row;
        thread_data[t].cols_a = cols_a;
        thread_data[t].cols_b = cols_b;
        thread_data[t].a = a;
        thread_data[t].b = b;
        thread_data[t].c = c;
        
        pthread_create(&threads[t], NULL, matrix_multiply_thread, &thread_data[t]);
        
        start_row = end_row;
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

// Thread function for elementwise operations
typedef struct {
    int start_idx;
    int end_idx;
    double *a;
    double *b;
    double *result;
    int operation; // 0: add, 1: subtract, 2: multiply
} elementwise_thread_data_t;

void* elementwise_operation_thread(void *arg) {
    elementwise_thread_data_t *data = (elementwise_thread_data_t*)arg;
    
    switch(data->operation) {
        case 0: // Add
            for (int i = data->start_idx; i < data->end_idx; i++) {
                data->result[i] = data->a[i] + data->b[i];
            }
            break;
        case 1: // Subtract
            for (int i = data->start_idx; i < data->end_idx; i++) {
                data->result[i] = data->a[i] - data->b[i];
            }
            break;
        case 2: // Multiply
            for (int i = data->start_idx; i < data->end_idx; i++) {
                data->result[i] = data->a[i] * data->b[i];
            }
            break;
    }
    
    pthread_exit(NULL);
}

// Parallel elementwise operations
void tensor_elementwise_parallel(int size, double *a, double *b, double *result, int operation, int num_threads) {
    pthread_t threads[num_threads];
    elementwise_thread_data_t thread_data[num_threads];
    
    int elems_per_thread = size / num_threads;
    int remainder = size % num_threads;
    
    int start_idx = 0;
    for (int t = 0; t < num_threads; t++) {
        int end_idx = start_idx + elems_per_thread + (t < remainder ? 1 : 0);
        
        thread_data[t].start_idx = start_idx;
        thread_data[t].end_idx = end_idx;
        thread_data[t].a = a;
        thread_data[t].b = b;
        thread_data[t].result = result;
        thread_data[t].operation = operation;
        
        pthread_create(&threads[t], NULL, elementwise_operation_thread, &thread_data[t]);
        
        start_idx = end_idx;
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

// Wrapper functions
void tensor_add_parallel(int size, double *a, double *b, double *result, int num_threads) {
    tensor_elementwise_parallel(size, a, b, result, 0, num_threads);
}

void tensor_sub_parallel(int size, double *a, double *b, double *result, int num_threads) {
    tensor_elementwise_parallel(size, a, b, result, 1, num_threads);
}

void tensor_mul_elementwise_parallel(int size, double *a, double *b, double *result, int num_threads) {
    tensor_elementwise_parallel(size, a, b, result, 2, num_threads);
}

// Batch processing with parallelism for neural networks
typedef struct {
    int start_sample;
    int end_sample;
    int input_dim;
    int output_dim;
    double *inputs;
    double *weights;
    double *biases;
    double *outputs;
} batch_thread_data_t;

void* process_batch_thread(void *arg) {
    batch_thread_data_t *data = (batch_thread_data_t*)arg;
    
    for (int i = data->start_sample; i < data->end_sample; i++) {
        // For each sample in this thread's portion of the batch
        for (int j = 0; j < data->output_dim; j++) {
            double sum = data->biases[j];
            for (int k = 0; k < data->input_dim; k++) {
                sum += data->inputs[i * data->input_dim + k] * data->weights[k * data->output_dim + j];
            }
            // Apply ReLU
            data->outputs[i * data->output_dim + j] = sum > 0 ? sum : 0;
        }
    }
    
    pthread_exit(NULL);
}

// Process a batch of samples through a layer in parallel
void process_batch_parallel(int batch_size, int input_dim, int output_dim, 
                           double *inputs, double *weights, double *biases, 
                           double *outputs, int num_threads) {
    pthread_t threads[num_threads];
    batch_thread_data_t thread_data[num_threads];
    
    int samples_per_thread = batch_size / num_threads;
    int remainder = batch_size % num_threads;
    
    int start_sample = 0;
    for (int t = 0; t < num_threads; t++) {
        int end_sample = start_sample + samples_per_thread + (t < remainder ? 1 : 0);
        
        thread_data[t].start_sample = start_sample;
        thread_data[t].end_sample = end_sample;
        thread_data[t].input_dim = input_dim;
        thread_data[t].output_dim = output_dim;
        thread_data[t].inputs = inputs;
        thread_data[t].weights = weights;
        thread_data[t].biases = biases;
        thread_data[t].outputs = outputs;
        
        pthread_create(&threads[t], NULL, process_batch_thread, &thread_data[t]);
        
        start_sample = end_sample;
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}