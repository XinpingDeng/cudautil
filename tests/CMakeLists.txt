cmake_minimum_required(VERSION 3.13)

project(tests LANGUAGES CUDA CXX C)

include(CheckLanguage)
check_language(CUDA)

set(UTIL_INC ../util/inc)
set(CUDA_SAMPLES_COMMON_INC /usr/local/cuda/samples/common/inc)
set(Catch2WithMain Catch2::Catch2WithMain)

add_executable(test_operators test_operators.cu)
target_include_directories(test_operators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(test_operators PRIVATE Catch2WithMain util)
set_target_properties(test_operators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_generators test_generators.cu)
target_include_directories(test_generators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(test_generators PRIVATE Catch2WithMain util curand)
set_target_properties(test_generators PROPERTIES LINKER_LANGUAGE CUDA)
