#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Store N element-wise results in output; this kernel returns no value.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
    
}