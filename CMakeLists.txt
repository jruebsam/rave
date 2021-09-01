cmake_minimum_required(VERSION 3.28.1)
project(RaVe LANGUAGES CUDA CXX)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

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

target_link_libraries(rave PRIVATE glfw glm)
target_include_directories(rave PUBLIC inc ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(rave PUBLIC ${glfw3_INCLUDE_DIRS})

set_property(TARGET rave PROPERTY CUDA_ARCHITECTURES 86)
set_property(TARGET rave PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_custom_command(
    TARGET rave POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink              
        ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:rave>/shaders
    COMMENT "Creating symlink to shader directory" VERBATIM
)
