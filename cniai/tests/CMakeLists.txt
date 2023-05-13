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

add_executable(test_nvjpeg_decoder test_nvjpeg_decoder.cc ${NVJPEG_SRC} ${COMMON_SRC} ${UTIL_SRC})
target_link_libraries(test_nvjpeg_decoder ${CUDA_LIBS} ${GFLAGS_LIBRARIES})

add_executable(test_thread_pool test_thread_pool.cc ${COMMON_SRC})
target_link_libraries(test_thread_pool pthread)

add_executable(test_log test_log.cc ${COMMON_SRC})
target_link_libraries(test_log ${GFLAGS_LIBRARIES})

add_executable(test_cuda_kernel_rgb_packed_planar_swap test_cuda_kernel_rgb_packed_planar_swap.cc)
target_link_libraries(test_cuda_kernel_rgb_packed_planar_swap ${CUDA_LIBS} cniai_cuda_kernel)