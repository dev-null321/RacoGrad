#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Matrix multiplication kernel as a string
const char* matrixMulKernelSource = 
"__kernel void matrixMul(\n"
"   const int M, const int N, const int K,\n"
"   __global const double* A,\n" 
"   __global const double* B,\n" 
"   __global double* C) {\n"
"   \n"
"   // Get global position in the grid\n"
"   const int row = get_global_id(0);\n" 
"   const int col = get_global_id(1);\n"
"   \n"
"   // Ensure we don't go out of bounds\n"
"   if (row < M && col < N) {\n"
"       double sum = 0.0;\n"
"       for (int k = 0; k < K; k++) {\n"
"           sum += A[row * K + k] * B[k * N + col];\n"
"       }\n"
"       C[row * N + col] = sum;\n"
"   }\n"
"}\n";

// OpenCL context and related variables
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_int err;
cl_bool is_opencl_initialized = CL_FALSE;

// Initialize OpenCL
cl_bool initialize_opencl() {
    if (is_opencl_initialized) return CL_TRUE;
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (err != CL_SUCCESS) {
        printf("Failed to get platform ID: %d\n", err);
        return CL_FALSE;
    }
    
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS) {
        // Try CPU if GPU is not available
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        if (err != CL_SUCCESS) {
            printf("Failed to get device ID: %d\n", err);
            return CL_FALSE;
        }
        printf("Using CPU device (GPU not available)\n");
    }
    
    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create context: %d\n", err);
        return CL_FALSE;
    }
    
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create command queue: %d\n", err);
        clReleaseContext(context);
        return CL_FALSE;
    }
    
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&matrixMulKernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program: %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return CL_FALSE;
    }
    
    // Build the program executable
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        
        printf("Failed to build program executable: %d\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return CL_FALSE;
    }
    
    // Create the compute kernel
    kernel = clCreateKernel(program, "matrixMul", &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel: %d\n", err);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return CL_FALSE;
    }
    
    is_opencl_initialized = CL_TRUE;
    return CL_TRUE;
}

// Clean up OpenCL resources
void cleanup_opencl() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    is_opencl_initialized = CL_FALSE;
}

// Matrix multiplication using OpenCL
int matrix_multiply_opencl(int M, int N, int K, double* A, double* B, double* C) {
    cl_mem d_A, d_B, d_C;
    size_t global_work_size[2];
    const size_t A_size = M * K * sizeof(double);
    const size_t B_size = K * N * sizeof(double);
    const size_t C_size = M * N * sizeof(double);
    
    // Initialize OpenCL if not already done
    if (!initialize_opencl()) {
        return -1;
    }
    
    // Create memory buffers on the device
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_size, A, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create buffer for A: %d\n", err);
        return -1;
    }
    
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B_size, B, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create buffer for B: %d\n", err);
        clReleaseMemObject(d_A);
        return -1;
    }
    
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, C_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create buffer for C: %d\n", err);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        return -1;
    }
    
    // Set the arguments of the kernel
    err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
    err |= clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
    err |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_A);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_B);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_C);
    
    if (err != CL_SUCCESS) {
        printf("Failed to set kernel arguments: %d\n", err);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        return -1;
    }
    
    // Define work group size and execute the kernel
    global_work_size[0] = M;
    global_work_size[1] = N;
    
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to execute kernel: %d\n", err);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        return -1;
    }
    
    // Read back the results from the device
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, C_size, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to read output matrix: %d\n", err);
        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        return -1;
    }
    
    // Release resources
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    
    return 0;
}

// Expose a function that can be called to check if OpenCL is available
int check_opencl_available() {
    cl_bool result = initialize_opencl();
    if (result == CL_TRUE) {
        cleanup_opencl();
        return 1;
    }
    return 0;
}