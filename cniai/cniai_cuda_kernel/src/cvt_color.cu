//
// Created by abel on 23-2-19.
//

#include "cniai_cuda_kernel/preprocess.h"


namespace cniai {
namespace preprocess {


__global__ void rgb_packed_planar_swap_kernel(const char * src, char* dst, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        int src_idx = (x + y * width) * 3;
        int dst_idx = x + y * width;
        dst[dst_idx + 0 * width * height] = src[src_idx + 0];
        dst[dst_idx + 1 * width * height] = src[src_idx + 1];
        dst[dst_idx + 2 * width * height] = src[src_idx + 2];
    }
}

void rgb_packed_planar_swap(const char* src, char* dst, int width, int height, cudaStream_t cudaStream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgb_packed_planar_swap_kernel<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(src, dst, width, height);
}


}
}
