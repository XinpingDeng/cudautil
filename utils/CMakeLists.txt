cmake_minimum_required(VERSION 3.13)

project(libutils LANGUAGES CUDA CXX C)

find_library(HDF5_LIBRARY hdf5)

add_library(utils STATIC hdf5_utils.c) # The performance is much better, default to STATIC

#add_library(utils SHARED utils.cu) # The performance is really bad
#set_target_properties(utils PROPERTIES PUBLIC_HEADER "include/utils.h;include/defs.h")
target_include_directories(utils PRIVATE ${HDF5_INCLUDE_DIRS})
