//
// Created by abel on 23-2-19.
//

#include "cniai_cuda_kernel/preprocess.h"


namespace cniai {
namespace preprocess {


__global__ void rgbPackedPlanarSwapKernel(const char * src, char* dst, const int width, const int height) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        unsigned int srcIdx = (x + y * width) * 3;
        unsigned int dstIdx = x + y * width;
        dst[dstIdx + 0 * width * height] = src[srcIdx + 0];
        dst[dstIdx + 1 * width * height] = src[srcIdx + 1];
        dst[dstIdx + 2 * width * height] = src[srcIdx + 2];
    }
}


void rgbPackedPlanarSwap(const char* src, char* dst, const int width, const int height, cudaStream_t cudaStream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbPackedPlanarSwapKernel<<<numBlocks, threadsPerBlock, 0, cudaStream>>>(src, dst, width, height);
}


}
}
