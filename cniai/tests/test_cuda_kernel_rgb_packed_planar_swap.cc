//
// Created by abel on 23-2-25.
//

#include <cniai_cuda_kernel/preprocess.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cassert>


int main() {
    cudaStream_t cudaStream;
    cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);

    int imgWidth = 4;
    int imgHeight = 3;
    size_t imgSize = imgWidth * imgHeight * 3 * sizeof(uint8_t);

    void* srcHostImg;
    cudaMallocHost(&srcHostImg, imgSize);
    void* srcDeviceImg;
    cudaMalloc(&srcDeviceImg, imgSize);

    auto* srcHostImgUint8 = (uint8_t*)srcHostImg;
    for (int i = 0; i < imgSize; ++i) {
        srcHostImgUint8[i] = i % 3 + 1;
        if ((i % 3) == 0 && i != 0) {
            std::cout << " ";
        }
        if (i % (imgWidth * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(srcHostImgUint8[i]);
    }
    cudaMemcpy(srcDeviceImg, srcHostImg, imgSize, cudaMemcpyHostToDevice);

    std::cout << std::endl << std::endl;

    void* dstHostImg;
    cudaMallocHost(&dstHostImg, imgSize);

    void* dstDeviceImg;
    cudaMalloc(&dstDeviceImg, imgSize);
    cudaMemset(dstDeviceImg, 0, imgSize);

    cniai::preprocess::rgbPackedPlanarSwap(
            (char*)srcDeviceImg, (char*)dstDeviceImg, imgWidth, imgHeight, cudaStream);
    cudaStreamSynchronize(cudaStream);

    cudaMemcpy(dstHostImg, dstDeviceImg, imgSize, cudaMemcpyDeviceToHost);

    auto* dstHostImgUint8 = (uint8_t*)dstHostImg;
    for (int i = 0; i < imgSize; ++i) {
        if (i % (imgWidth * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(dstHostImgUint8[i]);
    }

    for (int i = 0; i < imgHeight; ++i) {
        for (int j = 0; j < imgWidth; ++j) {
            assert(dstHostImgUint8[i * imgWidth * 3 + j] == i + 1);
        }
    }

    cudaFreeHost(srcHostImg);
    cudaFree(srcDeviceImg);
    cudaFreeHost(dstHostImg);
    cudaFree(dstDeviceImg);
    cudaStreamDestroy(cudaStream);
    return 0;
}