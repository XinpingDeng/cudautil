cmake_minimum_required (VERSION 3.5)

project (cudautil LANGUAGES CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib) # For shared library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib) # For static library

set(CUDA_LIBRARIES cufft curand cudart)

add_executable(test tests/test.cu)
target_include_directories(test PRIVATE /usr/local/cuda-11/samples/common/inc)
target_link_libraries(test ${CUDA_LIBRARIES})
