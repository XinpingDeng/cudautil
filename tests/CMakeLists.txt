add_executable(test_operators test_operators.cu)

add_executable(test_generators test_generators.cu)
target_link_libraries(test_generators PRIVATE curand cpgplot)

add_executable(test_convertors test_convertors.cu)
target_link_libraries(test_convertors PRIVATE curand)

add_executable(test_mean_stddev_calculators test_mean_stddev_calculators.cu)
target_link_libraries(test_mean_stddev_calculators PRIVATE curand)

add_executable(test_differentiators test_differentiators.cu)
target_link_libraries(test_differentiators PRIVATE curand)

add_executable(test_doctest test_doctest.cu)
target_link_libraries(test_doctest PRIVATE curand)

add_executable(test_dada_header test_dada_header.c dada_header.c)
target_link_libraries(test_dada_header m ${PSRDADA_LIBRARY})

