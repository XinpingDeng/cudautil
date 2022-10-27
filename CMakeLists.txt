cmake_minimum_required(VERSION 3.13)

project(utils LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
endif()

set(CMAKE_LIBRARY_PATH $ENV{INSTALL_ROOT}/lib)
find_library(PSRDADA_LIBRARY psrdada)

include_directories(.)

add_subdirectory(tests)
add_subdirectory(utils)

# cmake -DCMAKE_INSTALL_PREFIX:PATH=/data/BALDR_0/den15c/ ..
