//
// Created by abel on 23-2-27.
//

#include "cniai_cuda_kernel/preprocess.h"
#include "cniai_cuda_kernel/common.h"


namespace cniai {
namespace preprocess {

template<int c = 3, bool is_output_planar = false, bool is_swap_r_b = false>
__global__ void rgb_resize_bilinear_pad_norm_kernel(const uint8_t *src, float *dst,
                                               const int src_width, const int src_height,
                                               const int img_width, const int img_height,
                                               const int dst_width, const int dst_height,
                                               const int img_x, const int img_y,
                                               const int pad0, const int pad1, const int pad2,
                                               float scale,
                                               float mean0, float mean1, float mean2,
                                               float std0, float std1, float std2,
                                               const float scale_x, const float scale_y) {
    const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
        return;

    float src_x = (dst_x - img_x) * scale_x;
    float src_y = (dst_y - img_y) * scale_y;

    bool is_in_img = img_y <= dst_y && dst_y < img_y + img_height && img_x <= dst_x && dst_x < img_x + img_width;
    for (int c_idx = 0; c_idx < c; c_idx++) {
        float out = 0;
        if (is_in_img) {
            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;
            const int x2_read = min(x2, src_width - 1);
            const int y2_read = min(y2, src_height - 1);

            uint8_t src_reg = src[y1 * src_width * c + x1 * c + c_idx];
            out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

            src_reg = src[y1 * src_width * c + x2_read * c + c_idx];
            out = out + src_reg * ((src_x - x1) * (y2 - src_y));

            src_reg = src[y2_read * src_width * c + x1 * c + c_idx];
            out = out + src_reg * ((x2 - src_x) * (src_y - y1));

            src_reg = src[y2_read * src_width * c + x2_read * c + c_idx];
            out = out + src_reg * ((src_x - x1) * (src_y - y1));

            float mean = c_idx == 0 ? mean0 : c_idx == 1 ? mean1 : mean2;
            float std = c_idx == 0 ? std0 : c_idx == 1 ? std1 : std2;

            out = (out * scale - mean) * std;
        } else {
            out = c_idx == 0 ? pad0 : c_idx == 1 ? pad1 : pad2;
        }


        int cur_c_idx = c_idx;
        if (is_swap_r_b) {
            cur_c_idx = c_idx == 0 ? 2 : c_idx == 2 ? 0 : 1;
        }

        int dst_current_idx = is_output_planar ?
                              dst_width * dst_height * cur_c_idx + dst_y * dst_width + dst_x :
                              dst_y * dst_width * c + dst_x * c + cur_c_idx;

        dst[dst_current_idx] = out;

    }
}


void rgb_resize_bilinear_pad_norm(const uint8_t *src, float *dst,
                                                    const int src_width, const int src_height,
                                                    const int img_width, const int img_height,
                                                    const int dst_width, const int dst_height,
                                                    const int img_x, const int img_y,
                                                    const int pad0, const int pad1, const int pad2,
                                                    float scale,
                                                    float mean0, float mean1, float mean2,
                                                    float std0, float std1, float std2, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / img_width;
    float scale_y = static_cast<float>(src_height) / img_height;

    rgb_resize_bilinear_pad_norm_kernel<3, false><<<grid, block, 0, cudaStream>>>(
                                                        src, dst, src_width, src_height,
                                                        img_width, img_height,
                                                        dst_width, dst_height, img_x, img_y,
                                                        pad0, pad1, pad2,
                                                        scale,
                                                        mean0, mean1, mean2,
                                                        std0, std1, std2,
                                                        scale_x, scale_y);
}


void rgb_resize_bilinear_pad_norm_output_planar(const uint8_t *src, float *dst,
                                             const int src_width, const int src_height,
                                             const int img_width, const int img_height,
                                             const int dst_width, const int dst_height,
                                             const int img_x, const int img_y,
                                             const int pad0, const int pad1, const int pad2,
                                             float scale,
                                             float mean0, float mean1, float mean2,
                                             float std0, float std1, float std2, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / img_width;
    float scale_y = static_cast<float>(src_height) / img_height;

    rgb_resize_bilinear_pad_norm_kernel<3, true><<<grid, block, 0, cudaStream>>>(
                                                        src, dst, src_width, src_height,
                                                        img_width, img_height,
                                                        dst_width, dst_height, img_x, img_y,
                                                        pad0, pad1, pad2,
                                                        scale,
                                                        mean0, mean1, mean2,
                                                        std0, std1, std2, scale_x, scale_y);
}


void rgb2bgr_resize_bilinear_pad_norm(const uint8_t *src, float *dst,
                                  const int src_width, const int src_height,
                                  const int img_width, const int img_height,
                                  const int dst_width, const int dst_height,
                                  const int img_x, const int img_y,
                                  const int pad0, const int pad1, const int pad2,
                                  float scale,
                                  float mean0, float mean1, float mean2,
                                  float std0, float std1, float std2, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / img_width;
    float scale_y = static_cast<float>(src_height) / img_height;

    rgb_resize_bilinear_pad_norm_kernel<3, false, true><<<grid, block, 0, cudaStream>>>(
            src, dst, src_width, src_height,
            img_width, img_height,
            dst_width, dst_height, img_x, img_y,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2,
            scale_x, scale_y);
}


void rgb2bgr_resize_bilinear_pad_norm_output_planar(const uint8_t *src, float *dst,
                                                const int src_width, const int src_height,
                                                const int img_width, const int img_height,
                                                const int dst_width, const int dst_height,
                                                const int img_x, const int img_y,
                                                const int pad0, const int pad1, const int pad2,
                                                float scale,
                                                float mean0, float mean1, float mean2,
                                                float std0, float std1, float std2, cudaStream_t cudaStream) {
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    float scale_x = static_cast<float>(src_width) / img_width;
    float scale_y = static_cast<float>(src_height) / img_height;

    rgb_resize_bilinear_pad_norm_kernel<3, true, true><<<grid, block, 0, cudaStream>>>(
            src, dst, src_width, src_height,
            img_width, img_height,
            dst_width, dst_height, img_x, img_y,
            pad0, pad1, pad2,
            scale,
            mean0, mean1, mean2,
            std0, std1, std2, scale_x, scale_y);
}

}
}