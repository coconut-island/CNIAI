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

void rgb_resize_bilinear_pad_norm(const uint8_t *src, float *dst,
                                  const int src_width, const int src_height,
                                  const int img_width, const int img_height,
                                  const int dst_width, const int dst_height,
                                  const int img_x, const int img_y,
                                  const int pad0, const int pad1, const int pad2,
                                  float scale,
                                  float mean0, float mean1, float mean2,
                                  float std0, float std1, float std2, cudaStream_t cudaStream);

void rgb_resize_bilinear_pad_norm_output_planar(const uint8_t *src, float *dst,
                                                const int src_width, const int src_height,
                                                const int img_width, const int img_height,
                                                const int dst_width, const int dst_height,
                                                const int img_x, const int img_y,
                                                const int pad0, const int pad1, const int pad2,
                                                float scale,
                                                float mean0, float mean1, float mean2,
                                                float std0, float std1, float std2, cudaStream_t cudaStream);

/**
 *
 * @param src src image packed format
 * @param dst dst image packed format
 * @param src_width
 * @param src_height
 * @param img_width resized image width
 * @param img_height resized image height
 * @param dst_width dst image width
 * @param dst_height  dst image height
 * @param img_x resized image start x in dst image
 * @param img_y resized image start y in dst image
 * @param pad0 src channel 0 pad, if src is rgb, here is r. if src is bgr, here is b.
 * @param pad1 src channel 1 pad
 * @param pad2 src channel 2 pad
 * @param scale
 * @param mean0 src channel 0 mean
 * @param mean1 src channel 1 mean
 * @param mean2 src channel 2 mean
 * @param std0 src channel 0 std
 * @param std1 src channel 1 std
 * @param std2 src channel 2 std
 * @param cudaStream
 */
void rgb2bgr_resize_bilinear_pad_norm(const uint8_t *src, float *dst,
                                      const int src_width, const int src_height,
                                      const int img_width, const int img_height,
                                      const int dst_width, const int dst_height,
                                      const int img_x, const int img_y,
                                      const int pad0, const int pad1, const int pad2,
                                      float scale,
                                      float mean0, float mean1, float mean2,
                                      float std0, float std1, float std2, cudaStream_t cudaStream);

/**
 *
 * @param src src image packed format
 * @param dst dst image planar format
 * @param src_width
 * @param src_height
 * @param img_width resized image width
 * @param img_height resized image height
 * @param dst_width dst image width
 * @param dst_height  dst image height
 * @param img_x resized image start x in dst image
 * @param img_y resized image start y in dst image
 * @param pad0 src channel 0 pad, if src is rgb, here is r. if src is bgr, here is b.
 * @param pad1 src channel 1 pad
 * @param pad2 src channel 2 pad
 * @param scale
 * @param mean0 src channel 0 mean
 * @param mean1 src channel 1 mean
 * @param mean2 src channel 2 mean
 * @param std0 src channel 0 std
 * @param std1 src channel 1 std
 * @param std2 src channel 2 std
 * @param cudaStream
 */
void rgb2bgr_resize_bilinear_pad_norm_output_planar(const uint8_t *src, float *dst,
                                                    const int src_width, const int src_height,
                                                    const int img_width, const int img_height,
                                                    const int dst_width, const int dst_height,
                                                    const int img_x, const int img_y,
                                                    const int pad0, const int pad1, const int pad2,
                                                    float scale,
                                                    float mean0, float mean1, float mean2,
                                                    float std0, float std1, float std2, cudaStream_t cudaStream);

}
}

#endif //CNIAI_TESTS_PREPROCESS_H
