cmake_minimum_required(VERSION 3.13)

project(utils LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} LESS 3.18)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 
-gencode arch=compute_75,code=sm_75 
-gencode arch=compute_86,code=sm_86")
  else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
    endif()
  endif()
  
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount 64")
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount 128")
  
  enable_language(CUDA)
else()
  message(WARNING "CUDAToolkit not found, CUDA related code will not be compiled")
endif()

add_subdirectory(bench)
add_subdirectory(tests)

# cmake -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_ROOT ..

#cmake -DPSRDADA_ROOT=//data/FRIGG_2/den15c ..
#cmake -DPSRDADA_ROOT=//data/BALDR_1/den15c ..
