//
// Created by abel on 23-2-12.
//

#ifndef CNIAI_CNIAI_NVJPEG_H
#define CNIAI_CNIAI_NVJPEG_H

#include <nvjpeg.h>
#include <memory>
#include <vector>

#include "common/common.h"
#include "common/logging.h"
#include "common/thread_pool.h"



namespace cniai {

class CniaiNvjpegImage {

public:
    CniaiNvjpegImage() = default;
    CniaiNvjpegImage(void * device_channel_ptrs[4], int width, int height, nvjpegOutputFormat_t format);
    ~CniaiNvjpegImage();

private:
    void * device_channel_ptrs_[NVJPEG_MAX_COMPONENT]{nullptr};
    void * host_data_ptr_ = nullptr;
    int width_{};
    int height_{};
    nvjpegOutputFormat_t format_{};

public:
    void * GetDeviceChannelPtr(int idx);
    void * GetHostDataPtr();
    int GetWidth() const;
    int GetHeight() const;
    nvjpegOutputFormat_t GetFormat();
    size_t size();
};


constexpr int pipeline_stages = 2;

struct decode_per_thread_params {
    cudaStream_t stream;
    nvjpegJpegState_t dec_state_cpu;
    nvjpegJpegState_t dec_state_gpu;
    nvjpegBufferPinned_t pinned_buffers[pipeline_stages];
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t  jpeg_streams[pipeline_stages];
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_dec_cpu;
    nvjpegJpegDecoder_t nvjpeg_dec_gpu;
};


// here have two optimization point
// 1. support NVJPEG_BACKEND_HARDWARE. if use A100, A30, H100. I don't have money!
class CniaiNvjpegDecoder {

public:
    explicit CniaiNvjpegDecoder(size_t thread_pool_count);
    ~CniaiNvjpegDecoder();

private:
    bool hw_decode_available_{}; // support in the future, if necessary

    ThreadPool workers_;

    cudaStream_t global_stream_{};

    nvjpegHandle_t nvjpeg_handle_{};

    std::vector<decode_per_thread_params> nvjpeg_per_thread_data_{};

public:
    std::shared_ptr<CniaiNvjpegImage> DecodeJpeg(const uint8_t *src_jpeg, size_t length, nvjpegOutputFormat_t output_format);

    std::vector<std::shared_ptr<CniaiNvjpegImage>> DecodeJpegBatch(const uint8_t *const *src_jpegs, const size_t *lengths, size_t image_size, nvjpegOutputFormat_t output_format);

private:
    static int dev_malloc(void **p, size_t s);

    static int dev_free(void *p);

    static int host_malloc(void** p, size_t s, unsigned int f);

    static int host_free(void* p);

};

}


#endif //CNIAI_CNIAI_NVJPEG_H
