cmake_minimum_required(VERSION 3.13)

project(tests LANGUAGES CUDA CXX C)

include(CheckLanguage)
check_language(CUDA)

set(UTIL_INC ../utilities)
set(CUDA_SAMPLES_COMMON_INC /usr/local/cuda/samples/common/inc)

add_executable(operators test_operators.cu)
target_include_directories(operators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
set_target_properties(operators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(generators test_generators.cu)
target_include_directories(generators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(generators PRIVATE curand cpgplot)
set_target_properties(generators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(convertors test_convertors.cu)
target_include_directories(convertors PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(convertors PRIVATE curand)
set_target_properties(convertors PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(mean_stddev_calculators test_mean_stddev_calculators.cu)
target_include_directories(mean_stddev_calculators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(mean_stddev_calculators PRIVATE curand)
set_target_properties(mean_stddev_calculators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(differentiators test_differentiators.cu)
target_include_directories(differentiators PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(differentiators PRIVATE curand)
set_target_properties(differentiators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(doctest test_doctest.cu)
target_include_directories(doctest PRIVATE ${UTIL_INC} ${CUDA_SAMPLES_COMMON_INC})
target_link_libraries(doctest PRIVATE curand)
set_target_properties(doctest PROPERTIES LINKER_LANGUAGE CUDA)
