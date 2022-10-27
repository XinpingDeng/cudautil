file(GLOB SRCS CONFIGURE_DEPENDS "*.cu"  "*.cpp" "*.c")
file(GLOB HDRS CONFIGURE_DEPENDS "*.cuh" "*.hpp" "*.h")
file(GLOB EXES CONFIGURE_DEPENDS "*.py"  "*.sh")

#add_library(utils STATIC hdf5_utils.c dada_utils.c shared_utils.cpp dada_cuda_utils.cu) # The performance is much better, default to STATIC
add_library(utils STATIC ${SRCS}) # The performance is much better, default to STATIC
set_target_properties(utils PROPERTIES PUBLIC_HEADER "${HDRS}")

install (TARGETS utils
  PUBLIC_HEADER DESTINATION $ENV{INSTALL_ROOT}/include/utils
  LIBRARY DESTINATION       $ENV{INSTALL_ROOT}/lib
  ARCHIVE DESTINATION       $ENV{INSTALL_ROOT}/lib)

install(FILES ${EXES} DESTINATION $ENV{INSTALL_ROOT}/bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE  # 755
)
