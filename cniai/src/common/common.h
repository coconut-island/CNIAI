//
// Created by abel on 23-2-17.
//

#ifndef CNIAI_COMMON_H
#define CNIAI_COMMON_H


#define CHECK_CUDA(call)                                                          \
    {                                                                             \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess)                                                    \
        {                                                                         \
            LOG_ERROR("CUDA Runtime failure: #{}:{}", _e, cudaGetErrorString(_e));\
            abort();                                                              \
        }                                                                         \
    }

#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            LOG_ERROR("NVJPEG Runtime failure: #{}", _e);                         \
            abort();                                                            \
        }                                                                       \
    }

#endif //CNIAI_COMMON_H
