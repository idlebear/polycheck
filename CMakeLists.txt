cmake_minimum_required(VERSION 3.6)
project(polycheck LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(Eigen3 3.4 REQUIRED)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ./pybind11/include
    ./include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_library(${PROJECT_NAME} SHARED
        src/point_in_polygon.cu
        include/point_in_polygon.h
        src/polycheck.cpp
        src/visibility.cu
        include/visibility.h
        include/common.h)

target_link_libraries(${PROJECT_NAME}
        ${PYTHON_LIBRARIES}
        cudart
        )

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
