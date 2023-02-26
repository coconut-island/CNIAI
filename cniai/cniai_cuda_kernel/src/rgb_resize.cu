//
// Created by abel on 23-2-26.
//

#include "cniai_cuda_kernel/preprocess.h"
#include "cniai_cuda_kernel/common.h"

namespace cniai {
namespace preprocess {


template<int c = 3, bool is_output_planar>
__global__ void rgb_resize_bilinear_kernel(const uint8_t *src, uint8_t *dst,
                                const int src_width, const int src_height,
                                const int dst_width, const int dst_height,
                                const float scale_x, const float scale_y) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
        return;

    float src_x = dst_x * scale_x;
    float src_y = dst_y * scale_y;

    for (int c_idx = 0; c_idx < c; c_idx++) {
        const int x1      = __float2int_rd(src_x);
        const int y1      = __float2int_rd(src_y);
        const int x2      = x1 + 1;
        const int y2      = y1 + 1;
        const int x2_read = min(x2, src_width - 1);
        const int y2_read = min(y2, src_height - 1);

        uint8_t out = 0;

        uint8_t src_reg = src[y1 * src_width * c + x1 * c + c_idx];
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = src[y1 * src_width * c + x2_read * c + c_idx];
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = src[y2_read * src_width * c + x1 * c + c_idx];
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = src[y2_read * src_width * c + x2_read * c + c_idx];
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        if (is_output_planar) {
            dst[dst_width * dst_height * c_idx + dst_y * dst_width + dst_x] = out;
        } else {
            dst[dst_y * dst_width * c + dst_x * c + c_idx] = out;
        }

    }
}

void rgb_resize_bilinear(const uint8_t *src, uint8_t *dst,
                         int src_width, int src_height,
                         int dst_width, int dst_height, cudaStream_t cudaStream) {

    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    rgb_resize_bilinear_kernel<3, false><<<grid, block, 0, cudaStream>>>(src, dst, src_width, src_height, dst_width, dst_height, scale_x, scale_y);
}

void rgb_resize_bilinear_output_planar(const uint8_t *src, uint8_t *dst,
                         int src_width, int src_height,
                         int dst_width, int dst_height, cudaStream_t cudaStream) {

    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    rgb_resize_bilinear_kernel<3, true><<<grid, block, 0, cudaStream>>>(src, dst, src_width, src_height, dst_width, dst_height, scale_x, scale_y);
}

}}