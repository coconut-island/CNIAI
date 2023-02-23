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
DEFINE_string(output_rgbi_bmp_path, "./1920x1080_rgbi.bmp", "Output rgbi bmp path.");
DEFINE_string(output_yu12_bmp_path, "./1920x1080_yu12.bmp", "Output yu12 bmp path.");


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

    auto jpeg_data = std::vector<char>(file_size);
    if (!input.read(jpeg_data.data(), file_size)) {
        std::cerr << "Cannot read from file: " << FLAGS_jpeg_path << std::endl;
        input.close();
        return EXIT_FAILURE;
    }

    // decode jpeg to interleaved RGB
    cniai::CniaiNvjpegImageDecoder cniai_nvjpeg_image_decoder(1);
    std::shared_ptr<cniai::CniaiNvjpegImage> decoded_rgbi_img = cniai_nvjpeg_image_decoder.DecodeJpeg((uint8_t*)jpeg_data.data(), file_size, NVJPEG_OUTPUT_RGBI);
    assert(decoded_rgbi_img != nullptr);
    int rgbi_width = decoded_rgbi_img->GetWidth();
    int rgbi_height = decoded_rgbi_img->GetHeight();
    auto rgbi_host_ptr = static_cast<const unsigned char *>(decoded_rgbi_img->GetHostDataPtr());
    cniai::image_util::writeBMPi(FLAGS_output_rgbi_bmp_path.c_str(), rgbi_host_ptr, rgbi_width, rgbi_height);


    // decode jpeg to yu12
    std::shared_ptr<cniai::CniaiNvjpegImage> decoded_yu12_img = cniai_nvjpeg_image_decoder.DecodeJpeg((uint8_t*)jpeg_data.data(), file_size, NVJPEG_OUTPUT_YUV);
    assert(decoded_yu12_img != nullptr);
    int yu12_width = decoded_yu12_img->GetWidth();
    int yu12_height = decoded_yu12_img->GetHeight();
    auto yu12host_ptr = static_cast<const unsigned char *>(decoded_yu12_img->GetHostDataPtr());
    cniai::image_util::writeYU12(FLAGS_output_yu12_bmp_path.c_str(), yu12host_ptr, yu12_width, yu12_height);

    gflags::ShutDownCommandLineFlags();
    return 0;
}