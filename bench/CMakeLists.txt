add_executable(bench_memory bench_memory.cu)
target_link_libraries(bench_memory curand)
target_include_directories(bench_memory PUBLIC ${CMAKE_SOURCE_DIR})
