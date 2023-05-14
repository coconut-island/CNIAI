//
// Created by abel on 23-5-14.
//

#ifndef CNIAI_CUDA_UTIL_H
#define CNIAI_CUDA_UTIL_H

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do {\
        cudaError_t _e = call;\
        if (_e != cudaSuccess) {\
            std::cerr << "CUDA error " << _e << ": " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    } while (0)
#endif  // CUDA_CHECK

#endif //CNIAI_CUDA_UTIL_H
