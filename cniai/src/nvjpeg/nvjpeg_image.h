//
// Created by abel on 23-5-13.
//

#ifndef CNIAI_NVJPEG_IMAGE_H
#define CNIAI_NVJPEG_IMAGE_H

#include <nvjpeg.h>


namespace cniai {


class NvjpegImage {

public:
    NvjpegImage() = default;
    NvjpegImage(void *deviceChannelPtrs[4], int width, int height, nvjpegOutputFormat_t format);
    ~NvjpegImage();

private:
    void *deviceChannelPtrs[NVJPEG_MAX_COMPONENT]{nullptr};
    void *hostDataPtr = nullptr;
    int width{};
    int height{};
    nvjpegOutputFormat_t format{};

public:
    void *getDeviceChannelPtr(int idx);
    void *getHostDataPtr();
    int getWidth() const;
    int getHeight() const;
    nvjpegOutputFormat_t getFormat();
    size_t size();
};


}


#endif //CNIAI_NVJPEG_IMAGE_H
