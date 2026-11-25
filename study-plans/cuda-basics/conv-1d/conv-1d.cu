#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {


    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int valuesCount = N - kN  + 1;

    if (idx < valuesCount) {
        float sum = 0.0f;

        for(int i = 0; i < kN; i++){
            sum += input[i + idx] * kernel[i];
        }

        output[idx] = sum;
    }
    
    

    

    
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
