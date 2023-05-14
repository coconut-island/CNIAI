//
// Created by abel on 23-2-12.
//

#include "nvjpeg_decoder.h"
#include "common/cuda_util.h"
#include "common/nvjpeg_util.h"
#include "common/logging.h"

#include <iostream>
#include <cassert>


namespace cniai {


int NvjpegDecoder::devMalloc(void **p, size_t s) {
    return static_cast<int>(cudaMalloc(p, s));
}

int NvjpegDecoder::devFree(void *p) {
    return static_cast<int>(cudaFree(p));
}

int NvjpegDecoder::hostMalloc(void** p, size_t s, unsigned int f) {
    return static_cast<int>(cudaHostAlloc(p, s, f));
}

int NvjpegDecoder::hostFree(void* p) {
    return static_cast<int>(cudaFreeHost(p));
}


float getScaleFactor(nvjpegChromaSubsampling_t chromaSubsampling) {
    float scaleFactor = 3.0; // it should be 3.0 for 444
    
    if (chromaSubsampling == NVJPEG_CSS_420 || chromaSubsampling == NVJPEG_CSS_411) {
        scaleFactor = 1.5;
    }

    if (chromaSubsampling == NVJPEG_CSS_422 || chromaSubsampling == NVJPEG_CSS_440) {
        scaleFactor = 2.0;
    }

    if (chromaSubsampling == NVJPEG_CSS_410) {
        scaleFactor = 1.25;
    }

    if (chromaSubsampling == NVJPEG_CSS_GRAY){
        scaleFactor = 1.0;
    }

    return scaleFactor;
}

bool pickGpuBackend(nvjpegJpegStream_t &jpegStream) {
    unsigned int frameWidth,frameHeight;
    nvjpegChromaSubsampling_t chromaSubsampling;

    NVJPEG_CHECK(nvjpegJpegStreamGetFrameDimensions(jpegStream,
                                                    &frameWidth, &frameHeight));
    NVJPEG_CHECK(nvjpegJpegStreamGetChromaSubsampling(jpegStream,&chromaSubsampling));
    auto scaleFactor = getScaleFactor(chromaSubsampling);

    bool useGpuBackend = false;
    // use NVJPEG_BACKEND_GPU_HYBRID when dimensions are greater than 512x512
    if (512 * 512 * 3 < static_cast<float>(frameHeight * frameWidth) * scaleFactor) {
        useGpuBackend = true;
    }
    return useGpuBackend;
}


NvjpegDecoder::NvjpegDecoder(size_t threadPoolCount)
    : workers(threadPoolCount) {
    assert(threadPoolCount > 0);

    CUDA_CHECK(cudaStreamCreateWithFlags(&globalStream, cudaStreamNonBlocking));

    nvjpegDevAllocator_t devAllocator = {&devMalloc, &devFree};
    nvjpegPinnedAllocator_t pinnedAllocator ={&hostMalloc, &hostFree};
    NVJPEG_CHECK(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &devAllocator,
                                           &pinnedAllocator,NVJPEG_FLAGS_DEFAULT,  &nvjpegHandle));

    nvjpegPerThreadData.resize(threadPoolCount);

    static_assert(PIPELINE_STAGES >= 2, "We need at least two stages in the pipeline to allow buffering of the states, "
                                       "so the re-allocations won't interfere with asynchronous execution.");
    for (auto &nvjpegData : nvjpegPerThreadData) {
        // stream for decoding
        CUDA_CHECK(cudaStreamCreateWithFlags(&nvjpegData.stream, cudaStreamNonBlocking));

        NVJPEG_CHECK(nvjpegDecoderCreate(nvjpegHandle, NVJPEG_BACKEND_HYBRID, &nvjpegData.nvjpegDecCpu));
        NVJPEG_CHECK(nvjpegDecoderCreate(nvjpegHandle, NVJPEG_BACKEND_GPU_HYBRID, &nvjpegData.nvjpegDecGpu));
        NVJPEG_CHECK(nvjpegDecoderStateCreate(nvjpegHandle, nvjpegData.nvjpegDecCpu, &nvjpegData.decStateCpu));
        NVJPEG_CHECK(nvjpegDecoderStateCreate(nvjpegHandle, nvjpegData.nvjpegDecGpu, &nvjpegData.decStateGpu));

        NVJPEG_CHECK(nvjpegBufferDeviceCreate(nvjpegHandle, nullptr, &nvjpegData.deviceBuffer));

        for (int i = 0; i < PIPELINE_STAGES; i++) {
            NVJPEG_CHECK(nvjpegBufferPinnedCreate(nvjpegHandle, nullptr, &nvjpegData.pinnedBuffers[i]));
            NVJPEG_CHECK(nvjpegJpegStreamCreate(nvjpegHandle, &nvjpegData.jpegStreams[i]));
        }
        NVJPEG_CHECK(nvjpegDecodeParamsCreate(nvjpegHandle, &nvjpegData.nvjpegDecodeParams));

        NVJPEG_CHECK(nvjpegStateAttachDeviceBuffer(nvjpegData.decStateCpu, nvjpegData.deviceBuffer));
        NVJPEG_CHECK(nvjpegStateAttachDeviceBuffer(nvjpegData.decStateGpu, nvjpegData.deviceBuffer));
    }

}

NvjpegDecoder::~NvjpegDecoder() {
    for (auto &nvjpegData : nvjpegPerThreadData) {
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(nvjpegData.nvjpegDecodeParams));

        for (int i = 0; i < PIPELINE_STAGES; i++) {
            NVJPEG_CHECK(nvjpegJpegStreamDestroy(nvjpegData.jpegStreams[i]));
            NVJPEG_CHECK(nvjpegBufferPinnedDestroy(nvjpegData.pinnedBuffers[i]));
        }
        NVJPEG_CHECK(nvjpegBufferDeviceDestroy(nvjpegData.deviceBuffer));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(nvjpegData.decStateCpu));
        NVJPEG_CHECK(nvjpegJpegStateDestroy(nvjpegData.decStateGpu));
        NVJPEG_CHECK(nvjpegDecoderDestroy(nvjpegData.nvjpegDecCpu));
        NVJPEG_CHECK(nvjpegDecoderDestroy(nvjpegData.nvjpegDecGpu));

        CUDA_CHECK(cudaStreamDestroy(nvjpegData .stream));
    }

    NVJPEG_CHECK(nvjpegDestroy(nvjpegHandle));

    CUDA_CHECK(cudaStreamDestroy(globalStream));
}


std::shared_ptr<NvjpegImage> NvjpegDecoder::decodeJpeg(const uint8_t *srcJpeg, size_t length) {
    return decodeJpegBatch(&srcJpeg, &length, 1, defaultOutputFormat)[0];
}


std::shared_ptr<NvjpegImage> NvjpegDecoder::decodeJpeg(const uint8_t* srcJpeg, size_t length, nvjpegOutputFormat_t outputFormat) {
    return decodeJpegBatch(&srcJpeg, &length, 1, outputFormat)[0];
}


std::vector<std::shared_ptr<NvjpegImage>> NvjpegDecoder::decodeJpegBatch(const uint8_t *const *srcJpegs, const size_t *lengths, size_t imageCount) {
    return decodeJpegBatch(srcJpegs, lengths, imageCount, defaultOutputFormat);
}


std::vector<std::shared_ptr<NvjpegImage>> NvjpegDecoder::decodeJpegBatch(const uint8_t *const *srcJpegs, const size_t *lengths, const size_t imageCount, nvjpegOutputFormat_t outputFormat)  {
    std::vector<int> imgWidths(imageCount);
    std::vector<int> imgHeights(imageCount);

    std::vector<std::shared_ptr<NvjpegImage>> decodedJpegImages;
    decodedJpegImages.resize(imageCount);

    // output buffers
    std::vector<nvjpegImage_t> iouts(imageCount);
    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> iszs(imageCount);

    for (int i = 0; i < imageCount; i++) {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            iouts[i].channel[c] = nullptr;
            iouts[i].pitch[c] = 0;
            iszs[i].pitch[c] = 0;
        }
    }

    int singleImgWidths[NVJPEG_MAX_COMPONENT];
    int singleImgHeights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    for (int i = 0; i < imageCount; ++i) {
        NVJPEG_CHECK(nvjpegGetImageInfo(
                nvjpegHandle, const_cast<unsigned char *>(srcJpegs[i]), lengths[i],
                &channels, &subsampling, singleImgWidths, singleImgHeights));

        imgWidths[i] = singleImgWidths[0];
        imgHeights[i] = singleImgHeights[0];

        LOG_DEBUG("Image is {} channels.", channels);
        for (int c = 0; c < channels; c++) {
            LOG_DEBUG("Channel #{} size {} x {}", c, singleImgWidths[c], singleImgHeights[c]);
        }

        switch (subsampling) {
            case NVJPEG_CSS_444:
                LOG_DEBUG("YUV 4:4:4 chroma subsampling");
                break;
            case NVJPEG_CSS_440:
                LOG_DEBUG("YUV 4:4:0 chroma subsampling");
                break;
            case NVJPEG_CSS_422:
                LOG_DEBUG("YUV 4:2:2 chroma subsampling");
                break;
            case NVJPEG_CSS_420:
                LOG_DEBUG("YUV 4:2:0 chroma subsampling");
                break;
            case NVJPEG_CSS_411:
                LOG_DEBUG("YUV 4:1:1 chroma subsampling");
                break;
            case NVJPEG_CSS_410:
                LOG_DEBUG("YUV 4:1:0 chroma subsampling");
                break;
            case NVJPEG_CSS_GRAY:
                LOG_DEBUG("Grayscale JPEG ");
                break;
            case NVJPEG_CSS_UNKNOWN:
            case NVJPEG_CSS_410V:
                LOG_DEBUG("Unknown chroma subsampling");
        }

        int mul = 1;
        // in the case of interleaved RGB output, write only to single channel, but
        // 3 samples at once
        if (outputFormat == NVJPEG_OUTPUT_RGBI || outputFormat == NVJPEG_OUTPUT_BGRI) {
            channels = 1;
            mul = 3;
        }
            // in the case of rgb create 3 buffers with sizes of original image
        if (outputFormat == NVJPEG_OUTPUT_RGB ||
            outputFormat == NVJPEG_OUTPUT_BGR) {
            channels = 3;
            singleImgWidths[1] = singleImgWidths[2] = singleImgWidths[0];
            singleImgHeights[1] = singleImgHeights[2] = singleImgHeights[0];
        }

        // malloc output buffer
        for (int c = 0; c < channels; c++) {
            int aw = mul * singleImgWidths[c];
            int ah = singleImgHeights[c];
            int sz = aw * ah;
            iouts[i].pitch[c] = aw;
            if (sz > iszs[i].pitch[c]) {
                if (iouts[i].channel[c]) {
                    CUDA_CHECK(cudaFree(iouts[i].channel[c]));
                }
                CUDA_CHECK(cudaMalloc((void**)&iouts[i].channel[c], sz));
                iszs[i].pitch[c] = sz;
            }
        }
    }

    std::vector<int> bufferIndices(workers.size(), 0);
    auto &_nvjpegHandle = this->nvjpegHandle;
    auto &_nvjpegPerThreadData = this->nvjpegPerThreadData;
    std::vector<std::future<std::shared_ptr<NvjpegImage>>> nvjpegImageFutures;
    nvjpegImageFutures.resize(imageCount);
    for (int i = 0; i < imageCount; i++) {
        nvjpegImageFutures[i] = workers.enqueue(
                [&_nvjpegHandle, &_nvjpegPerThreadData, &bufferIndices, &iouts, &lengths, &srcJpegs, &outputFormat, &imgWidths, &imgHeights](int threadIdx, int iIdx)
                         {
                             auto& perThreadParams = _nvjpegPerThreadData[threadIdx];

                             NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(perThreadParams.nvjpegDecodeParams, outputFormat));
                             NVJPEG_CHECK(nvjpegJpegStreamParse(_nvjpegHandle, srcJpegs[iIdx], lengths[iIdx],
                                                                0, 0, perThreadParams.jpegStreams[bufferIndices[threadIdx]]));
                             bool useGpuBackend = pickGpuBackend(perThreadParams.jpegStreams[bufferIndices[threadIdx]]);

                             nvjpegJpegDecoder_t &decoder       = useGpuBackend ? perThreadParams.nvjpegDecGpu : perThreadParams.nvjpegDecCpu;
                             nvjpegJpegState_t   &decoderState  = useGpuBackend ? perThreadParams.decStateGpu  : perThreadParams.decStateCpu;

                             NVJPEG_CHECK(nvjpegStateAttachPinnedBuffer(decoderState,
                                                                        perThreadParams.pinnedBuffers[bufferIndices[threadIdx]]));

                             NVJPEG_CHECK(nvjpegDecodeJpegHost(_nvjpegHandle, decoder, decoderState,
                                                               perThreadParams.nvjpegDecodeParams, perThreadParams.jpegStreams[bufferIndices[threadIdx]]));

                             CUDA_CHECK(cudaStreamSynchronize(perThreadParams.stream));

                             NVJPEG_CHECK(nvjpegDecodeJpegTransferToDevice(_nvjpegHandle, decoder, decoderState,
                                                                           perThreadParams.jpegStreams[bufferIndices[threadIdx]], perThreadParams.stream));

                             NVJPEG_CHECK(nvjpegDecodeJpegDevice(_nvjpegHandle, decoder, decoderState,
                                                                 &iouts[iIdx], perThreadParams.stream));

                             // switch pinned buffer in pipeline mode to avoid an extra sync
                             bufferIndices[threadIdx] = (bufferIndices[threadIdx] + 1) % PIPELINE_STAGES;

                             CUDA_CHECK(cudaStreamSynchronize(perThreadParams.stream));


                             auto iout = iouts[iIdx];
                             int width = imgWidths[iIdx];
                             int height = imgHeights[iIdx];

                             return std::make_shared<NvjpegImage>(reinterpret_cast<void **>(iout.channel), width, height, outputFormat);
                         }, i);
    }

    for (int i = 0; i < imageCount; ++i) {
        decodedJpegImages[i] = nvjpegImageFutures[i].get();
    }

    cudaStreamSynchronize(globalStream);

    return decodedJpegImages;
}


void NvjpegDecoder::setDefaultOutputFormat(nvjpegOutputFormat_t outputFormat) {
    defaultOutputFormat = outputFormat;
}


}



