cmake_minimum_required(VERSION 3.13)

project(lib LANGUAGES CUDA CXX C)

include(CheckLanguage)
check_language(CUDA)

set(UTIL_INC ../inc)
set(CUDA_SAMPLES_COMMON_INC /usr/local/cuda/samples/common/inc)

add_library(util util.cu util.cpp util.c)
target_include_directories(util PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})

