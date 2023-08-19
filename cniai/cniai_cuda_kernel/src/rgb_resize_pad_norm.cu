//
// Created by abel on 23-2-27.
//

#include "cniai_cuda_kernel/preprocess.h"
#include "cniai_cuda_kernel/common.h"


namespace cniai {
namespace preprocess {


template<int channel = 3, bool isOutputPlanar = false, bool isSwapRB = false>
__global__ void rgbResizeBilinearPadNormKernel(const uint8_t *src, float *dst,
                                               const int srcWidth, const int srcHeight,
                                               const int imgWidth, const int imgHeight,
                                               const int dstWidth, const int dstHeight,
                                               const int imgX, const int imgY,
                                               const int pad0, const int pad1, const int pad2,
                                               const float scale,
                                               const float mean0, const float mean1, const float mean2,
                                               const float std0, const float std1, const float std2,
                                               const float scaleX, const float scaleY) {
    const unsigned int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float srcX = static_cast<float>(dstX - imgX) * scaleX;
    float srcY = static_cast<float>(dstY - imgY) * scaleY;

    bool isInImg = imgY <= dstY && dstY < imgY + imgHeight && imgX <= dstX && dstX < imgX + imgWidth;
    for (int cIdx = 0; cIdx < channel; cIdx++) {
        float out = 0;
        if (isInImg) {
            const int x1 = __float2int_rd(srcX);
            const int y1 = __float2int_rd(srcY);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;
            const int x2Read = min(x2, srcWidth - 1);
            const int y2Read = min(y2, srcHeight - 1);

            uint8_t srcReg = src[y1 * srcWidth * channel + x1 * channel + cIdx];
            out = out + srcReg * ((x2 - srcX) * (y2 - srcY));

            srcReg = src[y1 * srcWidth * channel + x2Read * channel + cIdx];
            out = out + srcReg * ((srcX - x1) * (y2 - srcY));

            srcReg = src[y2Read * srcWidth * channel + x1 * channel + cIdx];
            out = out + srcReg * ((x2 - srcX) * (srcY - y1));

            srcReg = src[y2Read * srcWidth * channel + x2Read * channel + cIdx];
            out = out + srcReg * ((srcX - x1) * (srcY - y1));

            float mean = cIdx == 0 ? mean0 : cIdx == 1 ? mean1 : mean2;
            float std = cIdx == 0 ? std0 : cIdx == 1 ? std1 : std2;

            out = (out * scale - mean) * std;
        } else {
            out = cIdx == 0 ? pad0 : cIdx == 1 ? pad1 : pad2;
        }


        int curChannelIdx = cIdx;
        if (isSwapRB) {
            curChannelIdx = cIdx == 0 ? 2 : cIdx == 2 ? 0 : 1;
        }

        int dstCurrentIdx = isOutputPlanar ?
                              dstWidth * dstHeight * curChannelIdx + dstY * dstWidth + dstX :
                              dstY * dstWidth * channel + dstX * channel + curChannelIdx;

        dst[dstCurrentIdx] = out;

    }
}


void rgbResizeBilinearPadNorm(const uint8_t *src, float *dst,
                              const int srcWidth, const int srcHeight,
                              const int imgWidth, const int imgHeight,
                              const int dstWidth, const int dstHeight,
                              const int imgX, const int imgY,
                              const int pad0, const int pad1, const int pad2,
                              const float scale,
                              const float mean0, const float mean1, const float mean2,
                              const float std0, const float std1, const float std2,
                              cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadNormKernel<3, false><<<grid, block, 0, cudaStream>>>(
            src, dst,
            srcWidth, srcHeight,
            imgWidth, imgHeight,
            dstWidth, dstHeight,
            imgX, imgY,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2,
            scaleX, scaleY);
}


void rgbResizeBilinearPadNormOutputPlanar(const uint8_t *src, float *dst,
                                          const int srcWidth, const int srcHeight,
                                          const int imgWidth, const int imgHeight,
                                          const int dstWidth, const int dstHeight,
                                          const int imgX, const int imgY,
                                          const int pad0, const int pad1, const int pad2,
                                          const float scale,
                                          const float mean0, const float mean1, const float mean2,
                                          const float std0, const float std1, const float std2,
                                          cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadNormKernel<3, true><<<grid, block, 0, cudaStream>>>(
            src, dst,
            srcWidth, srcHeight,
            imgWidth, imgHeight,
            dstWidth, dstHeight,
            imgX, imgY,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2,
            scaleX, scaleY);
}


void rgbToBgrResizeBilinearPadNorm(const uint8_t *src, float *dst,
                                   const int srcWidth, const int srcHeight,
                                   const int imgWidth, const int imgHeight,
                                   const int dstWidth, const int dstHeight,
                                   const int imgX, const int imgY,
                                   const int pad0, const int pad1, const int pad2,
                                   const float scale,
                                   const float mean0, const float mean1, const float mean2,
                                   const float std0, const float std1, const float std2,
                                   cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadNormKernel<3, false, true><<<grid, block, 0, cudaStream>>>(
            src, dst,
            srcWidth, srcHeight,
            imgWidth, imgHeight,
            dstWidth, dstHeight,
            imgX, imgY,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2,
            scaleX, scaleY);
}


void rgbToBgrResizeBilinearPadNormOutputPlanar(const uint8_t *src, float *dst,
                                               const int srcWidth, const int srcHeight,
                                               const int imgWidth, const int imgHeight,
                                               const int dstWidth, const int dstHeight,
                                               const int imgX, const int imgY,
                                               const int pad0, const int pad1, const int pad2,
                                               const float scale,
                                               const float mean0, const float mean1, const float mean2,
                                               const float std0, const float std1, const float std2,
                                               cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
    float scaleX = static_cast<float>(srcWidth) / static_cast<float>(imgWidth);
    float scaleY = static_cast<float>(srcHeight) / static_cast<float>(imgHeight);

    rgbResizeBilinearPadNormKernel<3, true, true><<<grid, block, 0, cudaStream>>>(
            src, dst, srcWidth, srcHeight,
            imgWidth, imgHeight,
            dstWidth, dstHeight, imgX, imgY,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2,
            scaleX, scaleY);
}


}
}