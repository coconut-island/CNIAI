//
// Created by abel on 23-2-12.
//

#include "nvjpeg_decoder.h"

#include <iostream>


namespace cniai {


CniaiNvjpegImage::CniaiNvjpegImage(void * device_channel_ptrs[NVJPEG_MAX_COMPONENT], int width, int height, nvjpegOutputFormat_t format)
    : width_(width), height_(height) {

    for (int i = 0; i < NVJPEG_MAX_COMPONENT; ++i) {
        device_channel_ptrs_[i] = device_channel_ptrs[i];
    }

    switch (format) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
        case NVJPEG_OUTPUT_YUV:
        case NVJPEG_OUTPUT_Y:
            format_ = format;
            break;
        case NVJPEG_OUTPUT_UNCHANGED:
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(format));
            abort();
    }
}

CniaiNvjpegImage::~CniaiNvjpegImage() {
    for (auto &c : device_channel_ptrs_) {
        if (c) CHECK_CUDA(cudaFree(c))
    }

    if (host_data_ptr_) {
        CHECK_CUDA(cudaFreeHost(host_data_ptr_))
    }
}

void* CniaiNvjpegImage::GetDeviceChannelPtr(int idx) {
    return device_channel_ptrs_[idx];
}

void* CniaiNvjpegImage::GetHostDataPtr() {
    if (host_data_ptr_) {
        return host_data_ptr_;
    }

    CHECK_CUDA(cudaMallocHost(&host_data_ptr_, size()))

    switch (format_) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
            CHECK_CUDA(cudaMemcpy(host_data_ptr_, GetDeviceChannelPtr(0), size(), cudaMemcpyDeviceToHost))
            break;
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            CHECK_CUDA(cudaMemcpy(host_data_ptr_,
                                  GetDeviceChannelPtr(0),width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(host_data_ptr_) + width_ * height_ * sizeof(uint8_t),
                                  GetDeviceChannelPtr(1),width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(host_data_ptr_) + width_ * height_ * sizeof(uint8_t),
                                  GetDeviceChannelPtr(2), width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            break;
        case NVJPEG_OUTPUT_YUV:
            CHECK_CUDA(cudaMemcpy(host_data_ptr_, GetDeviceChannelPtr(0),
                                       width_ * height_ * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(host_data_ptr_) + width_ * height_ * sizeof(uint8_t), GetDeviceChannelPtr(1),
                                  width_ * height_ / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            CHECK_CUDA(cudaMemcpy(static_cast<uint8_t*>(host_data_ptr_) + width_ * height_ * 5 / 4 * sizeof(uint8_t), GetDeviceChannelPtr(2),
                                  width_ * height_ / 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost))
            break;
        default:
            LOG_ERROR("not support the format! format = {}", static_cast<int>(format_));
            abort();
    }
    return host_data_ptr_;
}

int CniaiNvjpegImage::GetWidth() const {
    return width_;
}

int  CniaiNvjpegImage::GetHeight() const {
    return height_;
}

nvjpegOutputFormat_t CniaiNvjpegImage::GetFormat() {
    return format_;
}

size_t CniaiNvjpegImage::size() {
    switch (format_) {
        case NVJPEG_OUTPUT_RGBI:
        case NVJPEG_OUTPUT_BGRI:
        case NVJPEG_OUTPUT_RGB:
        case NVJPEG_OUTPUT_BGR:
            return width_ * height_ * 3 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_YUV:
            return width_ * height_ * 3 / 2 * sizeof(uint8_t);
        case NVJPEG_OUTPUT_Y:
            return width_ * height_;
        default:
            LOG_ERROR("not support the format to cal size! return size 0, format = {}", static_cast<int>(format_));
            abort();
    }
}


int CniaiNvjpegDecoder::dev_malloc(void **p, size_t s) {
    return static_cast<int>(cudaMalloc(p, s));
}

int CniaiNvjpegDecoder::dev_free(void *p) {
    return static_cast<int>(cudaFree(p));
}

int CniaiNvjpegDecoder::host_malloc(void** p, size_t s, unsigned int f) {
    return static_cast<int>(cudaHostAlloc(p, s, f));
}

int CniaiNvjpegDecoder::host_free(void* p) {
    return static_cast<int>(cudaFreeHost(p));
}


float get_scale_factor(nvjpegChromaSubsampling_t chroma_subsampling) {
    float scale_factor = 3.0; // it should be 3.0 for 444
    if(chroma_subsampling == NVJPEG_CSS_420 || chroma_subsampling == NVJPEG_CSS_411) {
        scale_factor = 1.5;
    }
    else if(chroma_subsampling == NVJPEG_CSS_422 || chroma_subsampling == NVJPEG_CSS_440) {
        scale_factor = 2.0;
    }
    else if(chroma_subsampling == NVJPEG_CSS_410) {
        scale_factor = 1.25;
    }
    else if(chroma_subsampling == NVJPEG_CSS_GRAY){
        scale_factor = 1.0;
    }

    return scale_factor;
}

bool pick_gpu_backend(nvjpegJpegStream_t&  jpeg_stream) {
    unsigned int frame_width,frame_height;
    nvjpegChromaSubsampling_t chroma_subsampling;

    CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(jpeg_stream,
                                                    &frame_width, &frame_height))
    CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(jpeg_stream,&chroma_subsampling))
    auto scale_factor = get_scale_factor(chroma_subsampling);

    bool use_gpu_backend = false;
    // use NVJPEG_BACKEND_GPU_HYBRID when dimensions are greater than 512x512
    if (512 * 512 * 3 < static_cast<float>(frame_height * frame_width) * scale_factor) {
        use_gpu_backend = true;
    }
    return use_gpu_backend;
}


CniaiNvjpegDecoder::CniaiNvjpegDecoder(size_t thread_pool_count)
    : workers_(thread_pool_count) {
    assert(thread_pool_count > 0);

    CHECK_CUDA(cudaStreamCreateWithFlags(&global_stream_, cudaStreamNonBlocking))

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                           &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &nvjpeg_handle_))

    nvjpeg_per_thread_data_.resize(thread_pool_count);

    static_assert(pipeline_stages >= 2, "We need at least two stages in the pipeline to allow buffering of the states, "
                                       "so the re-allocations won't interfere with asynchronous execution.");
    for (auto &nvjpeg_data : nvjpeg_per_thread_data_) {
        // stream for decoding
        CHECK_CUDA( cudaStreamCreateWithFlags(&nvjpeg_data.stream, cudaStreamNonBlocking))

        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_HYBRID, &nvjpeg_data.nvjpeg_dec_cpu))
        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_GPU_HYBRID, &nvjpeg_data.nvjpeg_dec_gpu))
        CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_data.nvjpeg_dec_cpu, &nvjpeg_data.dec_state_cpu))
        CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_data.nvjpeg_dec_gpu, &nvjpeg_data.dec_state_gpu))

        CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle_, nullptr, &nvjpeg_data.device_buffer))

        for(int i = 0; i < pipeline_stages; i++) {
            CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle_, nullptr, &nvjpeg_data.pinned_buffers[i]))
            CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &nvjpeg_data.jpeg_streams[i]))
        }
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_data.nvjpeg_decode_params))

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_data.dec_state_cpu, nvjpeg_data.device_buffer))
        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_data.dec_state_gpu, nvjpeg_data.device_buffer))
    }

}

CniaiNvjpegDecoder::~CniaiNvjpegDecoder() {
    for (auto &nvjpeg_data : nvjpeg_per_thread_data_) {
        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_data.nvjpeg_decode_params))

        for(int i = 0; i < pipeline_stages; i++) {
            CHECK_NVJPEG(nvjpegJpegStreamDestroy(nvjpeg_data.jpeg_streams[i]))
            CHECK_NVJPEG(nvjpegBufferPinnedDestroy(nvjpeg_data.pinned_buffers[i]))
        }
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(nvjpeg_data.device_buffer))
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_data.dec_state_cpu))
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_data.dec_state_gpu))
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_data.nvjpeg_dec_cpu))
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_data.nvjpeg_dec_gpu))

        CHECK_CUDA(cudaStreamDestroy(nvjpeg_data.stream))
    }

    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle_))

    CHECK_CUDA(cudaStreamDestroy(global_stream_))
}


std::shared_ptr<CniaiNvjpegImage> CniaiNvjpegDecoder::DecodeJpeg(const uint8_t* src_jpeg, size_t length, nvjpegOutputFormat_t output_format) {
    return DecodeJpegBatch(&src_jpeg, &length, 1, output_format)[0];
}


std::vector<std::shared_ptr<CniaiNvjpegImage>> CniaiNvjpegDecoder::DecodeJpegBatch(const uint8_t *const *src_jpegs, const size_t *lengths, const size_t image_count, nvjpegOutputFormat_t output_format)  {
    std::vector<int> img_widths(image_count);
    std::vector<int> img_heights(image_count);

    std::vector<std::shared_ptr<CniaiNvjpegImage>> decoded_jpeg_images;
    decoded_jpeg_images.resize(image_count);

    // output buffers
    std::vector<nvjpegImage_t> iouts(image_count);
    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> iszs(image_count);

    for (int i = 0; i < image_count; i++) {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            iouts[i].channel[c] = nullptr;
            iouts[i].pitch[c] = 0;
            iszs[i].pitch[c] = 0;
        }
    }

    int single_img_widths[NVJPEG_MAX_COMPONENT];
    int single_img_heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    for (int i = 0; i < image_count; ++i) {
        CHECK_NVJPEG(nvjpegGetImageInfo(
                nvjpeg_handle_, const_cast<unsigned char *>(src_jpegs[i]), lengths[i],
                &channels, &subsampling, single_img_widths, single_img_heights))

        img_widths[i] = single_img_widths[0];
        img_heights[i] = single_img_heights[0];

        LOG_DEBUG("Image is {} channels.", channels);
        for (int c = 0; c < channels; c++) {
            LOG_DEBUG("Channel #{} size {} x {}", c, single_img_widths[c], single_img_heights[c]);
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
        if (output_format == NVJPEG_OUTPUT_RGBI || output_format == NVJPEG_OUTPUT_BGRI) {
            channels = 1;
            mul = 3;
        }
            // in the case of rgb create 3 buffers with sizes of original image
        if (output_format == NVJPEG_OUTPUT_RGB ||
                output_format == NVJPEG_OUTPUT_BGR) {
            channels = 3;
            single_img_widths[1] = single_img_widths[2] = single_img_widths[0];
            single_img_heights[1] = single_img_heights[2] = single_img_heights[0];
        }

        // malloc output buffer
        for (int c = 0; c < channels; c++) {
            int aw = mul * single_img_widths[c];
            int ah = single_img_heights[c];
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

    std::vector<int> buffer_indices(workers_.size(), 0);
    auto &nvjpeg_per_thread_data = nvjpeg_per_thread_data_;
    auto &nvjpeg_handle = nvjpeg_handle_;
    std::vector<std::future<std::shared_ptr<CniaiNvjpegImage>>> cniai_jpeg_futures;
    cniai_jpeg_futures.resize(image_count);
    for (int i = 0; i < image_count; i++) {
        cniai_jpeg_futures[i] = workers_.enqueue(
                [&nvjpeg_handle, &nvjpeg_per_thread_data, &buffer_indices, &iouts, &lengths, &src_jpegs, &output_format, &img_widths, &img_heights](int thread_idx, int iidx)
                         {
                             auto& per_thread_params = nvjpeg_per_thread_data[thread_idx];

                             CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(per_thread_params.nvjpeg_decode_params, output_format))
                             CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, src_jpegs[iidx], lengths[iidx],
                                                                0, 0, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]))
                             bool use_gpu_backend = pick_gpu_backend(per_thread_params.jpeg_streams[buffer_indices[thread_idx]]);

                             nvjpegJpegDecoder_t& decoder =   use_gpu_backend ?  per_thread_params.nvjpeg_dec_gpu: per_thread_params.nvjpeg_dec_cpu;
                             nvjpegJpegState_t&   decoder_state = use_gpu_backend ? per_thread_params.dec_state_gpu:per_thread_params.dec_state_cpu;

                             CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
                                                                        per_thread_params.pinned_buffers[buffer_indices[thread_idx]]))

                             CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle, decoder, decoder_state,
                                                               per_thread_params.nvjpeg_decode_params, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]))

                             CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream))

                             CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle, decoder, decoder_state,
                                                                           per_thread_params.jpeg_streams[buffer_indices[thread_idx]], per_thread_params.stream))

                             CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle, decoder, decoder_state,
                                                                 &iouts[iidx], per_thread_params.stream))

                             // switch pinned buffer in pipeline mode to avoid an extra sync
                             buffer_indices[thread_idx] = (buffer_indices[thread_idx] + 1) % pipeline_stages;

                             CHECK_CUDA(cudaStreamSynchronize(per_thread_params.stream))


                             auto iout = iouts[iidx];
                             int width = img_widths[iidx];
                             int height = img_heights[iidx];

                             return std::make_shared<CniaiNvjpegImage>(reinterpret_cast<void **>(iout.channel), width, height, output_format);
                         }, i);
    }

    for (int i = 0; i < image_count; ++i) {
        decoded_jpeg_images[i] = cniai_jpeg_futures[i].get();
    }

    cudaStreamSynchronize(global_stream_);

    return decoded_jpeg_images;
}


}



