cmake_minimum_required (VERSION 3.5)

project (cudautil LANGUAGES CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib) # For shared library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib) # For static library

add_executable(test tests/test.cu)
#target_include_directories(test PRIVATE $ENV{CUDA_ROOT}/samples/common/inc)
target_include_directories(test PRIVATE /usr/local/cuda/samples/common/inc)

# we do not need to add cuda libraries if we link it with CUDA
set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
