cmake_minimum_required(VERSION 3.13)

project(cniai_tests)

set(CMAKE_CXX_STANDARD 14)


include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/local/tensorrt/include)
link_directories(/usr/local/tensorrt/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../install/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../install/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(CUDA_LIBS cudart nvjpeg)
set(TENSORRT_LIBS nvinfer nvonnxparser)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/nvjpeg NVJPEG_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/common COMMON_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/util UTIL_SRC)


include(../cmake/install_dependencies.cmake)

add_executable(cniaiNvjpegDecoderTest cniaiNvjpegDecoderTest.cc ${NVJPEG_SRC} ${COMMON_SRC} ${UTIL_SRC})
target_link_libraries(cniaiNvjpegDecoderTest ${CUDA_LIBS} ${GFLAGS_LIBRARIES})

add_executable(threadPoolTest threadPoolTest.cc ${COMMON_SRC})
target_link_libraries(threadPoolTest pthread)

add_executable(logTest logTest.cc ${COMMON_SRC})
target_link_libraries(logTest ${GFLAGS_LIBRARIES})

add_executable(tensorrtTest tensorrtTest.cc ${COMMON_SRC} ${NVJPEG_SRC})
target_link_libraries(tensorrtTest ${GFLAGS_LIBRARIES} ${CUDA_LIBS} ${TENSORRT_LIBS})

add_executable(cudaKernelRgbPackedPlanarSwapTest cuda_kernel/cudaKernelRgbPackedPlanarSwapTest.cc)
target_link_libraries(cudaKernelRgbPackedPlanarSwapTest ${CUDA_LIBS} cniai_cuda_kernel)