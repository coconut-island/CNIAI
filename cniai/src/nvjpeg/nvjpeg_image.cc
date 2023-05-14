//
// Created by abel on 23-5-13.
//

#include "nvjpeg_image.h"

#include <memory>

#include "common/cuda_util.h"
#include "common/logging.h"


namespace cniai {


NvjpegImage::NvjpegImage(void * deviceChannelPtrs[NVJPEG_MAX_COMPONENT], int width, int height, nvjpegOutputFormat_t format)
        : width(width), height(height) {

    for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i) {
        this->deviceChannelPtrs[i] = deviceChannelPtrs[i];
    }

    switch (format) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
        case NVJPEG_OUTPUT_YUV:
        case NVJPEG_OUTPUT_Y:
            this->format = format;
            break;
        case NVJPEG_OUTPUT_UNCHANGED:
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(format));
            abort();
    }
}

NvjpegImage::~NvjpegImage() {
    for (auto &c : deviceChannelPtrs) {
        if (c) CUDA_CHECK(cudaFree(c));
    }

    if (hostDataPtr) {
        CUDA_CHECK(cudaFreeHost(hostDataPtr));
    }
}

void* NvjpegImage::getDeviceChannelPtr(int idx) {
    return deviceChannelPtrs[idx];
}

void* NvjpegImage::getHostDataPtr() {
    if (hostDataPtr) {
        return hostDataPtr;
    }

    CUDA_CHECK(cudaMallocHost(&hostDataPtr, size()));

    switch (format) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
            CUDA_CHECK(cudaMemcpy(hostDataPtr, getDeviceChannelPtr(0), size(), cudaMemcpyDeviceToHost));
            break;
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            CUDA_CHECK(cudaMemcpy(hostDataPtr,
                                  getDeviceChannelPtr(0),width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(hostDataPtr) + width * height * sizeof(uint8_t),
                                  getDeviceChannelPtr(1),width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(hostDataPtr) + width * height * sizeof(uint8_t),
                                  getDeviceChannelPtr(2), width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            break;
        case NVJPEG_OUTPUT_YUV:
            CUDA_CHECK(cudaMemcpy(hostDataPtr, getDeviceChannelPtr(0),
                                  width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(hostDataPtr) + width * height * sizeof(uint8_t), getDeviceChannelPtr(1),
                                  width * height / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(hostDataPtr) + width * height * 5 / 4 * sizeof(uint8_t), getDeviceChannelPtr(2),
                                  width * height / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            break;
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(format));
            abort();
    }
    return hostDataPtr;
}

int NvjpegImage::getWidth() const {
    return width;
}

int NvjpegImage::getHeight() const {
    return height;
}

nvjpegOutputFormat_t NvjpegImage::getFormat() {
    return format;
}

size_t NvjpegImage::size() {
    switch (format) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            return width * height * 3 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_YUV:
            return width * height * 3 / 2 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_Y:
            return width * height;
        default:
            LOG_ERROR("not support the format to cal size! return size 0, format = {}", static_cast<int>(format));
            abort();
    }
}


}