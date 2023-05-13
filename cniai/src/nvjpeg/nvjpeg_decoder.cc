//
// Created by abel on 23-2-12.
//

#include "nvjpeg_decoder.h"

#include <iostream>


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

    CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(jpegStream,
                                                    &frameWidth, &frameHeight))
    CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(jpegStream,&chromaSubsampling))
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

    CHECK_CUDA(cudaStreamCreateWithFlags(&globalStream, cudaStreamNonBlocking))

    nvjpegDevAllocator_t devAllocator = {&devMalloc, &devFree};
    nvjpegPinnedAllocator_t pinnedAllocator ={&hostMalloc, &hostFree};
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &devAllocator,
                                           &pinnedAllocator,NVJPEG_FLAGS_DEFAULT,  &nvjpegHandle))

    nvjpegPerThreadData.resize(threadPoolCount);

    static_assert(PIPELINE_STAGES >= 2, "We need at least two stages in the pipeline to allow buffering of the states, "
                                       "so the re-allocations won't interfere with asynchronous execution.");
    for (auto &nvjpegData : nvjpegPerThreadData) {
        // stream for decoding
        CHECK_CUDA( cudaStreamCreateWithFlags(&nvjpegData.stream, cudaStreamNonBlocking))

        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpegHandle, NVJPEG_BACKEND_HYBRID, &nvjpegData.nvjpegDecCpu))
        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpegHandle, NVJPEG_BACKEND_GPU_HYBRID, &nvjpegData.nvjpegDecGpu))
        CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpegHandle, nvjpegData.nvjpegDecCpu, &nvjpegData.decStateCpu))
        CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpegHandle, nvjpegData.nvjpegDecGpu, &nvjpegData.decStateGpu))

        CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpegHandle, nullptr, &nvjpegData.deviceBuffer))

        for (int i = 0; i < PIPELINE_STAGES; i++) {
            CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpegHandle, nullptr, &nvjpegData.pinnedBuffers[i]))
            CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpegHandle, &nvjpegData.jpegStreams[i]))
        }
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpegHandle, &nvjpegData.nvjpegDecodeParams))

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpegData.decStateCpu, nvjpegData.deviceBuffer))
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpegData.decStateGpu, nvjpegData.deviceBuffer))
    }

}

NvjpegDecoder::~NvjpegDecoder() {
    for (auto &nvjpegData : nvjpegPerThreadData) {
        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpegData.nvjpegDecodeParams))

        for (int i = 0; i < PIPELINE_STAGES; i++) {
            CHECK_NVJPEG(nvjpegJpegStreamDestroy(nvjpegData.jpegStreams[i]))
            CHECK_NVJPEG(nvjpegBufferPinnedDestroy(nvjpegData.pinnedBuffers[i]))
        }
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(nvjpegData.deviceBuffer))
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpegData.decStateCpu))
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpegData.decStateGpu))
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpegData.nvjpegDecCpu))
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpegData.nvjpegDecGpu))

        CHECK_CUDA(cudaStreamDestroy(nvjpegData .stream))
    }

    CHECK_NVJPEG(nvjpegDestroy(nvjpegHandle))

    CHECK_CUDA(cudaStreamDestroy(globalStream))
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
        CHECK_NVJPEG(nvjpegGetImageInfo(
                nvjpegHandle, const_cast<unsigned char *>(srcJpegs[i]), lengths[i],
                &channels, &subsampling, singleImgWidths, singleImgHeights))

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
                    CHECK_CUDA(cudaFree(iouts[i].channel[c]))
                }
                CHECK_CUDA(cudaMalloc((void**)&iouts[i].channel[c], sz))
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

                             CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(perThreadParams.nvjpegDecodeParams, outputFormat))
                             CHECK_NVJPEG(nvjpegJpegStreamParse(_nvjpegHandle, srcJpegs[iIdx], lengths[iIdx],
                                                                0, 0, perThreadParams.jpegStreams[bufferIndices[threadIdx]]))
                             bool useGpuBackend = pickGpuBackend(perThreadParams.jpegStreams[bufferIndices[threadIdx]]);

                             nvjpegJpegDecoder_t &decoder       = useGpuBackend ? perThreadParams.nvjpegDecGpu : perThreadParams.nvjpegDecCpu;
                             nvjpegJpegState_t   &decoderState  = useGpuBackend ? perThreadParams.decStateGpu  : perThreadParams.decStateCpu;

                             CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoderState,
                                                                        perThreadParams.pinnedBuffers[bufferIndices[threadIdx]]))

                             CHECK_NVJPEG(nvjpegDecodeJpegHost(_nvjpegHandle, decoder, decoderState,
                                                               perThreadParams.nvjpegDecodeParams, perThreadParams.jpegStreams[bufferIndices[threadIdx]]))

                             CHECK_CUDA(cudaStreamSynchronize(perThreadParams.stream))

                             CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(_nvjpegHandle, decoder, decoderState,
                                                                           perThreadParams.jpegStreams[bufferIndices[threadIdx]], perThreadParams.stream))

                             CHECK_NVJPEG(nvjpegDecodeJpegDevice(_nvjpegHandle, decoder, decoderState,
                                                                 &iouts[iIdx], perThreadParams.stream))

                             // switch pinned buffer in pipeline mode to avoid an extra sync
                             bufferIndices[threadIdx] = (bufferIndices[threadIdx] + 1) % PIPELINE_STAGES;

                             CHECK_CUDA(cudaStreamSynchronize(perThreadParams.stream))


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



