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

#include "nvjpeg_image.h"


namespace cniai {


constexpr int PIPELINE_STAGES = 2;


struct DecodePerThreadParams {
    cudaStream_t stream;
    nvjpegJpegState_t decStateCpu;
    nvjpegJpegState_t decStateGpu;
    nvjpegBufferPinned_t pinnedBuffers[PIPELINE_STAGES];
    nvjpegBufferDevice_t deviceBuffer;
    nvjpegJpegStream_t  jpegStreams[PIPELINE_STAGES];
    nvjpegDecodeParams_t nvjpegDecodeParams;
    nvjpegJpegDecoder_t nvjpegDecCpu;
    nvjpegJpegDecoder_t nvjpegDecGpu;
};


// here have two optimization point
// 1. support NVJPEG_BACKEND_HARDWARE. if use A100, A30, H100. I don't have money!
class NvjpegDecoder {

public:
    explicit NvjpegDecoder(size_t threadPoolCount);
    ~NvjpegDecoder();

private:
    bool hwDecodeAvailable{}; // support in the future, if necessary

    ThreadPool workers;

    cudaStream_t globalStream{};

    nvjpegHandle_t nvjpegHandle{};

    std::vector<DecodePerThreadParams> nvjpegPerThreadData{};

    nvjpegOutputFormat_t defaultOutputFormat = NVJPEG_OUTPUT_RGBI;

public:
    std::shared_ptr<NvjpegImage> decodeJpeg(const uint8_t *srcJpeg, size_t length);

    std::shared_ptr<NvjpegImage> decodeJpeg(const uint8_t *srcJpeg, size_t length, nvjpegOutputFormat_t outputFormat);

    std::vector<std::shared_ptr<NvjpegImage>> decodeJpegBatch(const uint8_t *const *srcJpegs, const size_t *lengths, size_t imageCount);

    std::vector<std::shared_ptr<NvjpegImage>> decodeJpegBatch(const uint8_t *const *srcJpegs, const size_t *lengths, size_t imageCount, nvjpegOutputFormat_t outputFormat);

    void setDefaultOutputFormat(nvjpegOutputFormat_t outputFormat);

private:
    static int devMalloc(void **p, size_t s);

    static int devFree(void *p);

    static int hostMalloc(void** p, size_t s, unsigned int f);

    static int hostFree(void* p);

};


}


#endif //CNIAI_CNIAI_NVJPEG_H
