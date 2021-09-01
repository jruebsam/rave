cmake_minimum_required(VERSION 2.8.12)
project(RaVe LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(
    rave 
    src/main.cpp
    src/glad.cpp
    src/Shader.cpp
    src/Window.cpp
    src/Canvas.cpp
    src/State.cu
    src/Simulation.cu
)

set_property(TARGET rave PROPERTY CUDA_ARCHITECTURES 86)
set_property(TARGET rave PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(rave PUBLIC inc)
target_link_libraries(rave ${CONAN_LIBS})

target_include_directories(rave PUBLIC "/usr/local/cuda-11.4/include")
target_link_libraries(rave "/usr/local/cuda-11.4/lib64/libcudart.so")

add_custom_command(
    TARGET rave POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink              
        ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:rave>/shaders
    COMMENT "Creating symlink to shader directory" VERBATIM
)
