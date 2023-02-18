//
// Created by abel on 23-2-12.
//

#include "nvjpeg_decoder.h"

#include <iostream>


namespace cniai {


CniaiJpeg::CniaiJpeg(void *device_data, int width, int height, nvjpegOutputFormat_t format)
    : device_data_(device_data), width_(width), height_(height) {

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

void *CniaiJpeg::GetDeviceData() {
    return device_data_;
}

int CniaiJpeg::GetWidth() const {
    return width_;
}

int CniaiJpeg::GetHeight() const {
    return height_;
}

nvjpegOutputFormat_t CniaiJpeg::GetFormat() {
    return format_;
}

size_t CniaiJpeg::size() {
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


CniaiNvjpegDecoder::CniaiNvjpegDecoder(nvjpegOutputFormat_t output_format, size_t thread_pool_count)
    : workers_(thread_pool_count) {
    assert(thread_pool_count > 0);

    this->output_format_ = output_format;

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

std::shared_ptr<CniaiJpeg> CniaiNvjpegDecoder::DecodeJpeg(std::vector<char> &src_jpeg) {
    auto src_jpeg_warp = std::vector<std::vector<char>>{src_jpeg};
    auto decoded_img_warp = std::vector<std::shared_ptr<CniaiJpeg>>();
    return DecodeJpegBatch(src_jpeg_warp)[0];
}


std::vector<std::shared_ptr<CniaiJpeg>> CniaiNvjpegDecoder::DecodeJpegBatch(std::vector<std::vector<char>> &src_jpegs)  {
    auto src_jpeg_count = src_jpegs.size();
    std::vector<int> img_widths(src_jpeg_count);
    std::vector<int> img_heights(src_jpeg_count);

    std::vector<std::shared_ptr<CniaiJpeg>> decoded_imgs;
    decoded_imgs.clear();
    decoded_imgs.resize(src_jpeg_count);

    // output buffers
    std::vector<nvjpegImage_t> iouts(src_jpeg_count);
    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> iszs(src_jpeg_count);

    for (int i = 0; i < iouts.size(); i++) {
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
    for (int i = 0; i < src_jpeg_count; ++i) {
        CHECK_NVJPEG(nvjpegGetImageInfo(
                nvjpeg_handle_, (unsigned char *)src_jpegs[i].data(), src_jpegs[i].size(),
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
                return decoded_imgs;
        }

        int mul = 1;
        // in the case of interleaved RGB output, write only to single channel, but
        // 3 samples at once
        if (output_format_ == NVJPEG_OUTPUT_RGBI || output_format_ == NVJPEG_OUTPUT_BGRI) {
            channels = 1;
            mul = 3;
        }
            // in the case of rgb create 3 buffers with sizes of original image
        if (output_format_ == NVJPEG_OUTPUT_RGB ||
                output_format_ == NVJPEG_OUTPUT_BGR) {
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
    auto &output_format = output_format_;

    std::vector<std::future<std::shared_ptr<CniaiJpeg>>> cniai_jpeg_futures;
    cniai_jpeg_futures.resize(src_jpeg_count);
    for (int i = 0; i < src_jpeg_count; i++) {
        cniai_jpeg_futures[i] = workers_.enqueue(
                [&nvjpeg_handle, &nvjpeg_per_thread_data, &buffer_indices, &iouts, &src_jpegs, &output_format, &img_widths, &img_heights](int thread_idx, int iidx)
                         {
                             auto& per_thread_params = nvjpeg_per_thread_data[thread_idx];

                             CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(per_thread_params.nvjpeg_decode_params, output_format))
                             CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, (const unsigned char *)src_jpegs[iidx].data(), src_jpegs[iidx].size(),
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

                             std::shared_ptr<CniaiJpeg> cniai_jpeg = nullptr;

                             auto &iout = iouts[iidx];
                             int width = img_widths[iidx];
                             int height = img_heights[iidx];
                             if (output_format == NVJPEG_OUTPUT_RGBI || output_format == NVJPEG_OUTPUT_BGRI) {
                                 void* rgbi_device_ptr;
                                 CHECK_CUDA(cudaMallocAsync(&rgbi_device_ptr, width * height * 3 * sizeof(uint8_t), per_thread_params.stream))
                                 cudaMemcpyAsync(rgbi_device_ptr, iout.channel[0],
                                                 width * height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);

                                 cniai_jpeg = std::make_shared<CniaiJpeg>(rgbi_device_ptr, img_widths[0], img_heights[0], output_format);

                                 cudaStreamSynchronize(per_thread_params.stream);
                                 return cniai_jpeg;
                             }

                             if (output_format == NVJPEG_OUTPUT_RGB || output_format == NVJPEG_OUTPUT_BGR) {
                                 void* rgb_device_ptr;
                                 CHECK_CUDA(cudaMallocAsync(&rgb_device_ptr, width * height * 3 * sizeof(uint8_t), per_thread_params.stream))

                                 cudaMemcpyAsync(rgb_device_ptr, iout.channel[0],
                                                 width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);
                                 cudaMemcpyAsync(static_cast<uint8_t*>(rgb_device_ptr) + width * height * sizeof(uint8_t), iout.channel[1],
                                                 width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);
                                 cudaMemcpyAsync(static_cast<uint8_t*>(rgb_device_ptr) + width * height * sizeof(uint8_t) * 2, iout.channel[2],
                                                 width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);

                                 cniai_jpeg = std::make_shared<CniaiJpeg>(
                                         rgb_device_ptr, width, height, output_format);

                                 cudaStreamSynchronize(per_thread_params.stream);
                                 return cniai_jpeg;
                             }

                             if (output_format == NVJPEG_OUTPUT_YUV) {
                                 void* yu12_device_ptr;
                                 CHECK_CUDA(cudaMallocAsync(&yu12_device_ptr, width * height * 3 / 2 * sizeof(uint8_t), per_thread_params.stream))
                                 cudaMemcpyAsync(yu12_device_ptr, iout.channel[0],
                                                 width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);
                                 cudaMemcpyAsync(static_cast<uint8_t*>(yu12_device_ptr) + width * height * sizeof(uint8_t), iout.channel[1],
                                                 width * height / 2 / 2 * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);
                                 cudaMemcpyAsync(static_cast<uint8_t*>(yu12_device_ptr) + width * height * sizeof(uint8_t) + width * height / 2 / 2 * sizeof(uint8_t), iout.channel[2],
                                                 width * height / 2 / 2 * sizeof(uint8_t), cudaMemcpyDeviceToDevice, per_thread_params.stream);

                                 cniai_jpeg = std::make_shared<CniaiJpeg>(yu12_device_ptr, img_widths[0], img_heights[0], output_format);

                                 cudaStreamSynchronize(per_thread_params.stream);
                                 return cniai_jpeg;
                             }

                             LOG_ERROR("not support the format, return nullptr, format = {}", static_cast<int>(output_format));

                             return cniai_jpeg;
                         }, i);
    }

    for (int i = 0; i < cniai_jpeg_futures.size(); ++i) {
        decoded_imgs[i] = cniai_jpeg_futures[i].get();
    }
    cudaStreamSynchronize(global_stream_);

    for (auto & iout : iouts) {
        for (auto &c : iout.channel)
            if (c) CHECK_CUDA(cudaFree(c))
    }

    return decoded_imgs;
}


}



