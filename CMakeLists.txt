cmake_minimum_required(VERSION 3.13)

project(utils LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
endif()

find_library(PSRDADA_LIB psrdada $ENV{PSRDADA_ROOT}/lib)

include_directories(.)

add_subdirectory(tests)
add_subdirectory(utils)

# cmake -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_ROOT ..
