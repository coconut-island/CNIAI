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
set(TENSORRT_LIBS nvinfer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/nvjpeg NVJPEG_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/common COMMON_SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../src/util UTIL_SRC)


set(CNIAI_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../install)

find_package(gflags PATHS ${CNIAI_INSTALL_DIR} NO_DEFAULT_PATH)
if (gflags_FOUND)
    message(STATUS "GFLAGS FOUND")
    message(STATUS "GFLAGS DIR: ${gflags_DIR}")
    message(STATUS "GFLAGS VERSION: ${gflags_VERSION}")
else ()
    message(STATUS "GFLAGS NOT FOUND, WILL BUILD")
    execute_process(COMMAND bash build.sh gflags WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty)
    find_package(gflags PATHS ${CNIAI_INSTALL_DIR} NO_DEFAULT_PATH REQUIRED)
endif ()


find_package(spdlog PATHS ${CNIAI_INSTALL_DIR} NO_DEFAULT_PATH)
if (spdlog_FOUND)
    message(STATUS "SPDLOG FOUND")
    message(STATUS "SPDLOG DIR: ${spdlog_DIR}")
    message(STATUS "SPDLOG VERSION: ${spdlog_VERSION}")
else ()
    message(STATUS "SPDLOG NOT FOUND, WILL BUILD")
    execute_process(COMMAND bash build.sh spdlog WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty)
    find_package(spdlog PATHS ${CNIAI_INSTALL_DIR} NO_DEFAULT_PATH REQUIRED)
endif ()


add_executable(cniaiNvjpegDecoderTest cniaiNvjpegDecoderTest.cc ${NVJPEG_SRC} ${COMMON_SRC} ${UTIL_SRC})
target_link_libraries(cniaiNvjpegDecoderTest ${CUDA_LIBS} ${GFLAGS_LIBRARIES})

add_executable(threadPoolTest threadPoolTest.cc ${COMMON_SRC})
target_link_libraries(threadPoolTest pthread)

add_executable(logTest logTest.cc ${COMMON_SRC})
target_link_libraries(logTest ${GFLAGS_LIBRARIES})
