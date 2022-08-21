cmake_minimum_required(VERSION 3.13)

project(libutils LANGUAGES CUDA CXX C)

set(INCLUDE_UTILS $ENV{UTILS_ROOT}/include)

add_library(utils STATIC hdf5_utils.c) # The performance is much better, default to STATIC
#add_library(utils SHARED utils.cu) # The performance is really bad

target_include_directories(utils PRIVATE ${INCLUDE_UTILS})

install (TARGETS utils
         ARCHIVE DESTINATION $ENV{UTILS_ROOT}/lib
         LIBRARY DESTINATION $ENV{UTILS_ROOT}/lib
         RUNTIME DESTINATION $ENV{UTILS_ROOT}/bin)
