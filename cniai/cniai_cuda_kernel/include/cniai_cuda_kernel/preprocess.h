//
// Created by abel on 23-2-19.
//

#ifndef CNIAI_TESTS_PREPROCESS_H
#define CNIAI_TESTS_PREPROCESS_H

#include <cuda_runtime_api.h>

namespace cniai {
namespace preprocess {

void rgb_packed_planar_swap(const char* src, char* dst, int width, int height, cudaStream_t cudaStream);

}
}

#endif //CNIAI_TESTS_PREPROCESS_H
