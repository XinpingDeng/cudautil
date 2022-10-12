file(GLOB SRCS CONFIGURE_DEPENDS "*.cu" "*.cpp" "*.c")

#add_library(utils STATIC hdf5_utils.c dada_utils.c shared_utils.cpp dada_cuda_utils.cu) # The performance is much better, default to STATIC
add_library(utils STATIC ${SRCS}) # The performance is much better, default to STATIC

install (TARGETS utils
         ARCHIVE DESTINATION $ENV{UTILS_ROOT}/lib
         LIBRARY DESTINATION $ENV{UTILS_ROOT}/lib
         RUNTIME DESTINATION $ENV{UTILS_ROOT}/bin)
