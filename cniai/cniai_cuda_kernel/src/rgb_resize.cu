//
// Created by abel on 23-2-26.
//

#include "cniai_cuda_kernel/preprocess.h"
#include "cniai_cuda_kernel/common.h"


namespace cniai {
namespace preprocess {


template<int channel = 3, bool isOutputPlanar>
__global__ void rgbResizeBilinearKernel(const uint8_t *src, uint8_t *dst,
                                        const int srcWidth, const int srcHeight,
                                        const int dstWidth, const int dstHeight,
                                        const float scaleX, const float scaleY) {
    const unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float srcX = static_cast<float>(dstX) * scaleX;
    float srcY = static_cast<float>(dstY) * scaleY;

    for (int cIdx = 0; cIdx < channel; cIdx++) {
        const int x1      = __float2int_rd(srcX);
        const int y1      = __float2int_rd(srcY);
        const int x2      = x1 + 1;
        const int y2      = y1 + 1;
        const int x2Read = min(x2, srcWidth - 1);
        const int y2Read = min(y2, srcHeight - 1);

        uint8_t out = 0;

        uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
        out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

        srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
        out = out + srcReg * ((srcX - x1) * (y2 - srcY));

        srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
        out = out + srcReg * ((x2 - srcX) * (srcY - y1));

        srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
        out = out + srcReg * ((srcX - x1) * (srcY - y1));

        int dstCurrentIdx = isOutputPlanar ?
                              dstWidth * dstHeight * cIdx + dstY * dstWidth + dstX :
                              dstY * dstWidth * channel + dstX * channel + cIdx;

        dst[dstCurrentIdx] = out;

    }
}


void rgbResizeBilinear(const uint8_t *src, uint8_t *dst,
                       const int srcWidth, const int srcHeight,
                       const int dstWidth, const int dstHeight,
                       cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

    rgbResizeBilinearKernel<3, false><<<grid, block, 0, cudaStream>>>(
            src, dst, srcWidth, srcHeight, dstWidth, dstHeight, scaleX, scaleY);
}


void rgbResizeBilinearOutputPlanar(const uint8_t *src, uint8_t *dst,
                                   const int srcWidth, const int srcHeight,
                                   const int dstWidth, const int dstHeight,
                                   cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

    rgbResizeBilinearKernel<3, true><<<grid, block, 0, cudaStream>>>(
            src, dst, srcWidth, srcHeight, dstWidth, dstHeight, scaleX, scaleY);
}


}}