//
// Created by abel on 23-2-12.
//
#include <fstream>
#include <iostream>

#include "nvjpeg/nvjpeg_decoder.h"
#include "util/image_util.h"
#include "common/logging.h"

#include <gflags/gflags.h>


DEFINE_string(LOG_LEVEL, "trace", "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(LOG_PATTERN, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v", "Log pattern");

DEFINE_string(JPEG_PATH, "../../resources/1920x1080_yu12.jpg", "Input jpeg path.");
DEFINE_string(OUTPUT_RGBI_BMP_PATH, "./1920x1080_rgbi.bmp", "Output rgbi bmp path.");
DEFINE_string(OUTPUT_YU12_BMP_PATH, "./1920x1080_yu12.bmp", "Output yu12 bmp path.");


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_LOG_PATTERN);
    SET_LOG_LEVEL(FLAGS_LOG_LEVEL);

    std::ifstream input(FLAGS_JPEG_PATH.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
        std::cerr << "Cannot open image: " << FLAGS_JPEG_PATH << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    std::streamsize fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    auto jpegData = std::vector<char>(fileSize);
    if (!input.read(jpegData.data(), fileSize)) {
        std::cerr << "Cannot read from file: " << FLAGS_JPEG_PATH << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    // decode jpeg to interleaved RGB
    cniai::NvjpegDecoder nvjpegDecoder(1);
    nvjpegDecoder.setDefaultOutputFormat(NVJPEG_OUTPUT_YUV);
    std::shared_ptr<cniai::NvjpegImage> decodedRgbiImg = nvjpegDecoder.decodeJpeg((uint8_t*)jpegData.data(), fileSize, NVJPEG_OUTPUT_RGBI);
    assert(decodedRgbiImg != nullptr);
    int rgbiWidth = decodedRgbiImg->getWidth();
    int rgbiHeight = decodedRgbiImg->getHeight();
    auto rgbiHostPtr = static_cast<const unsigned char *>(decodedRgbiImg->getHostDataPtr());
    cniai::image_util::writeBMPi(FLAGS_OUTPUT_RGBI_BMP_PATH.c_str(), rgbiHostPtr, rgbiWidth, rgbiHeight);


    // decode jpeg to yu12
    std::shared_ptr<cniai::NvjpegImage> decodedYu12Img = nvjpegDecoder.decodeJpeg((uint8_t*)jpegData.data(), fileSize);
    assert(decodedYu12Img != nullptr);
    int yu12Width = decodedYu12Img->getWidth();
    int yu12Height = decodedYu12Img->getHeight();
    auto yu12HostPtr = static_cast<const unsigned char *>(decodedYu12Img->getHostDataPtr());
    cniai::image_util::writeYU12(FLAGS_OUTPUT_YU12_BMP_PATH.c_str(), yu12HostPtr, yu12Width, yu12Height);

    gflags::ShutDownCommandLineFlags();
    return 0;
}