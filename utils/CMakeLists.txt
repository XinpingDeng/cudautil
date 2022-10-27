file(GLOB SRCS CONFIGURE_DEPENDS "*.cu"  "*.cpp" "*.c")
file(GLOB HDRS CONFIGURE_DEPENDS "*.cuh" "*.hpp" "*.h")
file(GLOB EXES CONFIGURE_DEPENDS "*.py"  "*.sh")

#add_library(utils STATIC hdf5_utils.c dada_utils.c shared_utils.cpp dada_cuda_utils.cu) # The performance is much better, default to STATIC
add_library(utils STATIC ${SRCS}) # The performance is much better, default to STATIC
set_target_properties(utils PROPERTIES PUBLIC_HEADER "${HDRS}")

install (TARGETS utils
  PUBLIC_HEADER DESTINATION include/utils
  LIBRARY DESTINATION       lib
  ARCHIVE DESTINATION       lib)

install(FILES ${EXES} DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE  # 755
)
