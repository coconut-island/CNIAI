//
// Created by abel on 23-2-19.
//

#ifndef CNIAI_TESTS_PREPROCESS_H
#define CNIAI_TESTS_PREPROCESS_H

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cniai {
namespace preprocess {

void rgb_packed_planar_swap(const char* src, char* dst, int width, int height, cudaStream_t cudaStream);

void rgb_resize_bilinear(const uint8_t *src, uint8_t *dst,
                         const int src_width, const int src_height,
                         const int dst_width, const int dst_height, cudaStream_t cudaStream);

void rgb_resize_bilinear_output_planar(const uint8_t *src, uint8_t *dst,
                                        int src_width, int src_height,
                                        int dst_width, int dst_height, cudaStream_t cudaStream);

void rgb_resize_bilinear_pad(const uint8_t *src, uint8_t *dst,
                                    const int src_width, const int src_height,
                                    const int img_width, const int img_height,
                                    const int dst_width, const int dst_height,
                                    const int img_x, const int img_y,
                                    const int pad0, const int pad1, const int pad2, cudaStream_t cudaStream);

void rgb_resize_bilinear_pad_output_planar(const uint8_t *src, uint8_t *dst,
                                                  const int src_width, const int src_height,
                                                  const int img_width, const int img_height,
                                                  const int dst_width, const int dst_height,
                                                  const int img_x, const int img_y,
                                                  const int pad0, const int pad1, const int pad2, cudaStream_t cudaStream);

}
}

#endif //CNIAI_TESTS_PREPROCESS_H
