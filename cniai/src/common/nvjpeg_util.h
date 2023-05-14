//
// Created by abel on 23-5-14.
//

#ifndef CNIAI_NVJPEG_UTIL_H
#define CNIAI_NVJPEG_UTIL_H

#include <nvjpeg.h>

#ifndef NVJPEG_CHECK
#define NVJPEG_CHECK(call)\
    do {\
        nvjpegStatus_t _e = call;\
        if (_e != NVJPEG_STATUS_SUCCESS) {\
            std::cerr << "NVJPEG error " << _e << " at " << __FILE__ << ":" << __LINE__ << std::endl;;\
            abort();\
        }\
    } while (0)
#endif  // NVJPEG_CHECK

#endif //CNIAI_NVJPEG_UTIL_H
