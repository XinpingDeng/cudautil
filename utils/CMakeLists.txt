file(GLOB SRCS CONFIGURE_DEPENDS "*.cu" "*.cpp" "*.c")
file(GLOB HDRS CONFIGURE_DEPENDS "*.cuh" "*.hpp" "*.h")

#add_library(utils STATIC hdf5_utils.c dada_utils.c shared_utils.cpp dada_cuda_utils.cu) # The performance is much better, default to STATIC
add_library(utils STATIC ${SRCS}) # The performance is much better, default to STATIC
set_target_properties(utils PROPERTIES PUBLIC_HEADER "${HDRS}")

install (TARGETS utils
  PUBLIC_HEADER DESTINATION $ENV{LOCAL_INCLUDE}/utils
  ARCHIVE DESTINATION       $ENV{LOCAL_LIB}
  LIBRARY DESTINATION       $ENV{LOCAL_LIB}
  RUNTIME DESTINATION       $ENV{LOCAL_BIN})
