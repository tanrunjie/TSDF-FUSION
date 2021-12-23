PROJECT(mapping)

cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mapping main.cpp utils.hpp)

link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(mapping ${OpenCV_LIBS})