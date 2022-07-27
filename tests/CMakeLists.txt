cmake_minimum_required(VERSION 3.13)

project(test LANGUAGES CUDA CXX C)

set(UTIL_INC ../inc)
set(CUDA_SAMPLES_COMMON_INC /usr/local/cuda/samples/common/inc)

add_executable(test test.cu)
target_include_directories(test PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(test PRIVATE util)
set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
