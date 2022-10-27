set(INCLUDE_CURRENT ../utils)

add_executable(test_operators test_operators.cu)
target_include_directories(test_operators PRIVATE ${INCLUDE_CURRENT})
set_target_properties(test_operators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_generators test_generators.cu)
target_include_directories(test_generators PRIVATE ${INCLUDE_CURRENT})
target_link_libraries(test_generators PRIVATE curand cpgplot)
set_target_properties(test_generators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_convertors test_convertors.cu)
target_include_directories(test_convertors PRIVATE ${INCLUDE_CURRENT})
target_link_libraries(test_convertors PRIVATE curand)
set_target_properties(test_convertors PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_mean_stddev_calculators test_mean_stddev_calculators.cu)
target_include_directories(test_mean_stddev_calculators PRIVATE ${INCLUDE_CURRENT})
target_link_libraries(test_mean_stddev_calculators PRIVATE curand)
set_target_properties(test_mean_stddev_calculators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_differentiators test_differentiators.cu)
target_include_directories(test_differentiators PRIVATE ${INCLUDE_CURRENT})
target_link_libraries(test_differentiators PRIVATE curand)
set_target_properties(test_differentiators PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_doctest test_doctest.cu)
target_include_directories(test_doctest PRIVATE ${INCLUDE_CURRENT})
target_link_libraries(test_doctest PRIVATE curand)
set_target_properties(test_doctest PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_dada_header test_dada_header.c dada_header.c)
target_link_libraries(test_dada_header m ${PSRDADA_LIBRARY})

