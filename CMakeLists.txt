cmake_minimum_required(VERSION 3.27)
project(rvs_laboratories_9 LANGUAGES CXX CUDA)

set(LIBGPUTK_PATH "/usr/local/libgputk" CACHE STRING "Path to libgputk")

add_executable(dataset_generator dataset_generator/dataset_generator.cpp)

target_include_directories(dataset_generator PRIVATE ${LIBGPUTK_PATH}/include)
target_link_directories(dataset_generator PRIVATE ${LIBGPUTK_PATH}/lib/)
target_link_libraries(dataset_generator gputk)

add_executable(rvs_laboratories_9 rvs_laboratories_9/rvs_laboratories_9.cu)
