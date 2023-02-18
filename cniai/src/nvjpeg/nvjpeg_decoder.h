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

class CniaiJpeg {

public:
    CniaiJpeg() = delete;
    CniaiJpeg(void *device_data, int width, int height, nvjpegOutputFormat_t format);

private:
    void *device_data_ = nullptr;
    int width_{};
    int height_{};
    nvjpegOutputFormat_t format_{};

public:
    void* GetDeviceData();
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


class CniaiNvjpegDecoder {

public:
    explicit CniaiNvjpegDecoder(nvjpegOutputFormat_t output_format, size_t thread_pool_count);
    ~CniaiNvjpegDecoder();

private:
    bool hw_decode_available_{}; // support in the future, if necessary

    ThreadPool workers_;

    cudaStream_t global_stream_{};

    nvjpegHandle_t nvjpeg_handle_{};

    std::vector<decode_per_thread_params> nvjpeg_per_thread_data_;

    nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_RGBI;

public:
    std::shared_ptr<CniaiJpeg> DecodeJpeg(std::vector<char> &src_jpeg);

    std::vector<std::shared_ptr<CniaiJpeg>> DecodeJpegBatch(std::vector<std::vector<char>> &src_jpegs);

private:
    static int dev_malloc(void **p, size_t s);

    static int dev_free(void *p);

    static int host_malloc(void** p, size_t s, unsigned int f);

    static int host_free(void* p);

};

}


#endif //CNIAI_CNIAI_NVJPEG_H
