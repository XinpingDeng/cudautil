cmake_minimum_required(VERSION 3.13)

project(src LANGUAGES CUDA CXX C)

include_directories(../include)
add_library(util cudautil.cu util.c)
target_include_directories(util PRIVATE /usr/local/cuda/samples/common/inc)

