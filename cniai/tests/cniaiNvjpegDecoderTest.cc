//
// Created by abel on 23-2-12.
//
#include <fstream>
#include <iostream>

#include "nvjpeg/nvjpeg_decoder.h"
#include "util/image_util.h"

#include <gflags/gflags.h>

DEFINE_string(log_level, "trace", "Log level, includes [trace, debug, info, warn, err, critical, off]");
DEFINE_string(log_pattern, "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%@] %v", "Log pattern");

DEFINE_string(jpeg_path, "../../resources/1920x1080.jpg", "Input jpeg path.");
DEFINE_string(output_bmp_path, "./1920x1080.bmp", "Output bmp path.");


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SET_LOG_PATTERN(FLAGS_log_pattern);
    SET_LOG_LEVEL(FLAGS_log_level);

    std::ifstream input(FLAGS_jpeg_path.c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
        std::cerr << "Cannot open image: " << FLAGS_jpeg_path << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);

    auto jpeg_data = std::vector<char>();
    if (!input.read(jpeg_data.data(), file_size)) {
        std::cerr << "Cannot read from file: " << FLAGS_jpeg_path << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    cniai::CniaiNvjpegDecoder cniai_nvjpeg_decoder(NVJPEG_OUTPUT_RGBI, 6);
    std::shared_ptr<cniai::CniaiJpeg> decoded_img = cniai_nvjpeg_decoder.DecodeJpeg(jpeg_data);
    assert(decoded_img != nullptr);

    const auto *d_RGB = (const unsigned char *)decoded_img->GetDeviceData() ;
    int width = decoded_img->GetWidth();
    int height = decoded_img->GetHeight();

    std::vector<unsigned char> vchanRGB(decoded_img->size());
    unsigned char *chanRGB = vchanRGB.data();

    CHECK_CUDA(cudaMemcpy(chanRGB, d_RGB, decoded_img->size(), cudaMemcpyDeviceToHost))

    cniai::image_util::writeBMPi(FLAGS_output_bmp_path.c_str(), chanRGB, width, height);

    gflags::ShutDownCommandLineFlags();
    return 0;
}