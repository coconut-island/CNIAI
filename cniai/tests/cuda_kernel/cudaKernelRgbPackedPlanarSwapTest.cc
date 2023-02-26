//
// Created by abel on 23-2-25.
//

#include <cniai_cuda_kernel/preprocess.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <iostream>
#include <cassert>


int main() {

    cudaStream_t cudaStream;
    cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);

    int img_width = 4;
    int img_height = 3;
    size_t img_size = img_width * img_height * 3 * sizeof(uint8_t);

    void* src_host_img;
    cudaMallocHost(&src_host_img, img_size);
    void* src_device_img;
    cudaMalloc(&src_device_img, img_size);

    auto* src_host_img_uint8 = (uint8_t*)src_host_img;
    for (int i = 0; i < img_size; ++i) {
        src_host_img_uint8[i] = i % 3 + 1;
        if ((i % 3) == 0 && i != 0) {
            std::cout << " ";
        }
        if (i % (img_width * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(src_host_img_uint8[i]);
    }
    cudaMemcpy(src_device_img, src_host_img, img_size, cudaMemcpyHostToDevice);


    std::cout << std::endl << std::endl;

    void* dst_host_img;
    cudaMallocHost(&dst_host_img, img_size);

    void* dst_device_img;
    cudaMalloc(&dst_device_img, img_size);
    cudaMemset(dst_device_img, 0, img_size);

    cniai::preprocess::rgb_packed_planar_swap(
            (char*)src_device_img, (char*)dst_device_img, img_width, img_height, cudaStream);
    cudaStreamSynchronize(cudaStream);

    cudaMemcpy(dst_host_img, dst_device_img, img_size, cudaMemcpyDeviceToHost);

    auto* dst_host_img_uint8 = (uint8_t*)dst_host_img;
    for (int i = 0; i < img_size; ++i) {
        if (i % (img_width * 3) == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << std::to_string(dst_host_img_uint8[i]);
    }

    for (int i = 0; i < img_height; ++i) {
        for (int j = 0; j < img_width; ++j) {
            assert(dst_host_img_uint8[i * img_width * 3 + j] == i + 1);
        }
    }


    cudaFreeHost(src_host_img);
    cudaFree(src_device_img);
    cudaFreeHost(dst_host_img);
    cudaFree(dst_device_img);
    cudaStreamDestroy(cudaStream);
    return 0;
}