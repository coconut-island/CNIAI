cmake_minimum_required(VERSION 3.13)

project(cniai)

set(CMAKE_CXX_STANDARD 14)


include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/usr/local/tensorrt/include)
link_directories(/usr/local/tensorrt/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/install/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/install/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CUDA_LIBS cudart nvjpeg)
set(TENSORRT_LIBS nvinfer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/common SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/nvjpeg SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/util SRCS)

include(cmake/install_dependencies.cmake)

add_executable(cniai src/cniai.cc ${SRCS})
target_link_libraries(cniai PRIVATE ${CUDA_LIBS} ${TENSORRT_LIBS} ${GFLAGS_LIBRARIES})