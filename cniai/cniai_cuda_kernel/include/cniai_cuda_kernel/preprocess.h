//
// Created by abel on 23-2-19.
//

#ifndef CNIAI_CUDA_KERNEL_PREPROCESS_H
#define CNIAI_CUDA_KERNEL_PREPROCESS_H

#include <cuda_runtime_api.h>
#include <cstdint>


namespace cniai {
namespace preprocess {


void rgbPackedPlanarSwap(const char* src, char* dst,
                         const int width, const int height,
                         cudaStream_t cudaStream);

void rgbResizeBilinear(const uint8_t *src, uint8_t *dst,
                       const int srcWidth, const int srcHeight,
                       const int dstWidth, const int dstHeight,
                       cudaStream_t cudaStream);

void rgbResizeBilinearOutputPlanar(const uint8_t *src, uint8_t *dst,
                                   const int srcWidth, const int srcHeight,
                                   const int dstWidth, const int dstHeight,
                                   cudaStream_t cudaStream);

void rgbResizeBilinearPad(const uint8_t *src, uint8_t *dst,
                          const int srcWidth, const int srcHeight,
                          const int imgWidth, const int imgHeight,
                          const int dstWidth, const int dstHeight,
                          const int imgX, const int imgY,
                          const int pad0, const int pad1, const int pad2,
                          cudaStream_t cudaStream);

void rgbResizeBilinearPadOutputPlanar(const uint8_t *src, uint8_t *dst,
                                      const int srcWidth, const int srcHeight,
                                      const int imgWidth, const int imgHeight,
                                      const int dstWidth, const int dstHeight,
                                      const int imgX, const int imgY,
                                      const int pad0, const int pad1, const int pad2,
                                      cudaStream_t cudaStream);

void rgbResizeBilinearPadNorm(const uint8_t *src, float *dst,
                              const int srcWidth, const int srcHeight,
                              const int imgWidth, const int imgHeight,
                              const int dstWidth, const int dstHeight,
                              const int imgX, const int imgY,
                              const int pad0, const int pad1, const int pad2,
                              const float scale,
                              const float mean0, const float mean1, const float mean2,
                              const float std0, const float std1, const float std2,
                              cudaStream_t cudaStream);

void rgbResizeBilinearPadNormOutputPlanar(const uint8_t *src, float *dst,
                                          const int srcWidth, const int srcHeight,
                                          const int imgWidth, const int imgHeight,
                                          const int dstWidth, const int dstHeight,
                                          const int imgX, const int imgY,
                                          const int pad0, const int pad1, const int pad2,
                                          const float scale,
                                          const float mean0, const float mean1, const float mean2,
                                          const float std0, const float std1, const float std2,
                                          cudaStream_t cudaStream);

/**
 *
 * @param src src image packed format
 * @param dst dst image packed format
 * @param srcWidth
 * @param srcHeight
 * @param imgWidth resized image width
 * @param imgHeight resized image height
 * @param dstWidth dst image width
 * @param dstHeight  dst image height
 * @param imgX resized image start x in dst image
 * @param imgY resized image start y in dst image
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
void rgbToBgrResizeBilinearPadNorm(const uint8_t *src, float *dst,
                                   const int srcWidth, const int srcHeight,
                                   const int imgWidth, const int imgHeight,
                                   const int dstWidth, const int dstHeight,
                                   const int imgX, const int imgY,
                                   const int pad0, const int pad1, const int pad2,
                                   const float scale,
                                   const float mean0, const float mean1, const float mean2,
                                   const float std0, const float std1, const float std2,
                                   cudaStream_t cudaStream);

/**
 *
 * @param src src image packed format
 * @param dst dst image planar format
 * @param srcWidth
 * @param srcHeight
 * @param imgWidth resized image width
 * @param imgHeight resized image height
 * @param dstWidth dst image width
 * @param dstHeight  dst image height
 * @param imgX resized image start x in dst image
 * @param imgY resized image start y in dst image
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
void rgbToBgrResizeBilinearPadNormOutputPlanar(const uint8_t *src, float *dst,
                                               const int srcWidth, const int srcHeight,
                                               const int imgWidth, const int imgHeight,
                                               const int dstWidth, const int dstHeight,
                                               const int imgX, const int imgY,
                                               const int pad0, const int pad1, const int pad2,
                                               const float scale,
                                               const float mean0, const float mean1, const float mean2,
                                               const float std0, const float std1, const float std2,
                                               cudaStream_t cudaStream);


}
}


#endif //CNIAI_CUDA_KERNEL_PREPROCESS_H
