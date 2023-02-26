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

}
}

#endif //CNIAI_TESTS_PREPROCESS_H
