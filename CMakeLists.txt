cmake_minimum_required(VERSION 3.13)

project(utils LANGUAGES CUDA CXX C)

add_subdirectory(tests)
add_subdirectory(utils)

# header only so we only copy headers
set(INCLUDE_UTILS $ENV{UTILS_ROOT}/include)
install(DIRECTORY utils # source directory
        DESTINATION ${INCLUDE_UTILS} # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h" # select header files
)
