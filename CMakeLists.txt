cmake_minimum_required(VERSION 3.13)

project(utils LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
endif()

set(CMAKE_LIBRARY_PATH $ENV{LOCAL}/lib)
find_library(PSRDADA_LIBRARY psrdada)

add_subdirectory(tests)
add_subdirectory(utils)
