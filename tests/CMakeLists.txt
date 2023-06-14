find_library(PSRDADA psrdada ${PSRDADA_ROOT}/lib)

if(NOT PSRDADA)
  message(WARNING "PSRDADA not found, PSRDADA related code will not be compiled")
endif()

add_executable(test_operators test_operators.cu)
target_include_directories(test_operators PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

add_executable(test_generators test_generators.cu)
target_link_libraries(test_generators curand cpgplot)
target_include_directories(test_generators PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

add_executable(test_convertors test_convertors.cu)
target_link_libraries(test_convertors curand)
target_include_directories(test_convertors PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

add_executable(test_mean_stddev_calculators test_mean_stddev_calculators.cu)
target_link_libraries(test_mean_stddev_calculators curand)
target_include_directories(test_mean_stddev_calculators PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

add_executable(test_differentiators test_differentiators.cu)
target_link_libraries(test_differentiators curand)
target_include_directories(test_differentiators PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

add_executable(test_doctest test_doctest.cu)
target_link_libraries(test_doctest curand)
target_include_directories(test_doctest PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest)

if(PSRDADA)
  add_executable(test_dada_header test_dada_header.c dada_header.c)
  target_link_libraries(test_dada_header m ${PSRDADA})
  target_include_directories(test_dada_header PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/doctest ${PSRDADA_ROOT}/include)
endif()
