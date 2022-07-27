cmake_minimum_required(VERSION 3.13)

project(test LANGUAGES CUDA CXX C)

include_directories(../util/include) # so that I do not need to add it in my source code

add_executable(test test.cu)
target_include_directories(test PRIVATE /usr/local/cuda/samples/common/inc)
target_link_libraries(test PRIVATE util)

# we do not need to add cuda libraries if we link it with CUDA
set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
