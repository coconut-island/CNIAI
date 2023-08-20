//
// Created by abel on 23-5-13.
//

#include "nvjpeg_image.h"

#include <memory>

#include "common/cuda_util.h"
#include "common/logging.h"


namespace cniai {


NvjpegImage::NvjpegImage(void * deviceChannelPtrs[NVJPEG_MAX_COMPONENT], int width, int height, nvjpegOutputFormat_t format)
        : mWidth(width), mHeight(height) {

    for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i) {
        this->mDeviceChannelPtrs[i] = deviceChannelPtrs[i];
    }

    switch (format) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
        case NVJPEG_OUTPUT_YUV:
        case NVJPEG_OUTPUT_Y:
            this->mFormat = format;
            break;
        case NVJPEG_OUTPUT_UNCHANGED:
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(format));
            abort();
    }
}

NvjpegImage::~NvjpegImage() {
    for (auto &c : mDeviceChannelPtrs) {
        if (c) CUDA_CHECK(cudaFree(c));
    }

    if (mHostDataPtr) {
        CUDA_CHECK(cudaFreeHost(mHostDataPtr));
    }
}

void* NvjpegImage::getDeviceChannelPtr(int idx) {
    return mDeviceChannelPtrs[idx];
}

void* NvjpegImage::getHostDataPtr() {
    if (mHostDataPtr) {
        return mHostDataPtr;
    }

    CUDA_CHECK(cudaMallocHost(&mHostDataPtr, size()));

    switch (mFormat) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
            CUDA_CHECK(cudaMemcpy(mHostDataPtr, getDeviceChannelPtr(0), size(), cudaMemcpyDeviceToHost));
            break;
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            CUDA_CHECK(cudaMemcpy(mHostDataPtr,
                                  getDeviceChannelPtr(0),mWidth * mHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(mHostDataPtr) + mWidth * mHeight * sizeof(uint8_t),
                                  getDeviceChannelPtr(1),mWidth * mHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(mHostDataPtr) + mWidth * mHeight * sizeof(uint8_t),
                                  getDeviceChannelPtr(2), mWidth * mHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            break;
        case NVJPEG_OUTPUT_YUV:
            CUDA_CHECK(cudaMemcpy(mHostDataPtr, getDeviceChannelPtr(0),
                                  mWidth * mHeight * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(mHostDataPtr) + mWidth * mHeight * sizeof(uint8_t), getDeviceChannelPtr(1),
                                  mWidth * mHeight / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(static_cast<uint8_t*>(mHostDataPtr) + mWidth * mHeight * 5 / 4 * sizeof(uint8_t), getDeviceChannelPtr(2),
                                  mWidth * mHeight / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            break;
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(mFormat));
            abort();
    }
    return mHostDataPtr;
}

int NvjpegImage::getWidth() const {
    return mWidth;
}

int NvjpegImage::getHeight() const {
    return mHeight;
}

nvjpegOutputFormat_t NvjpegImage::getFormat() {
    return mFormat;
}

size_t NvjpegImage::size() {
    switch (mFormat) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            return mWidth * mHeight * 3 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_YUV:
            return mWidth * mHeight * 3 / 2 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_Y:
            return mWidth * mHeight;
        default:
            LOG_ERROR("not support the format to cal size! return size 0, format = {}", static_cast<int>(mFormat));
            abort();
    }
}


}