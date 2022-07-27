cmake_minimum_required (VERSION 3.5)

project (util LANGUAGES CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

add_executable(test tests/test.cu src/util.c src/cudautil.cu)
target_include_directories(test PRIVATE /usr/local/cuda/samples/common/inc)

# we do not need to add cuda libraries if we link it with CUDA
set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
