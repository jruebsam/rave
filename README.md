# RAVE - A solver for 2D Rayleigh Benard Convection on the GPU

This is a simple implementation of a solver for the incompressible navier-stokes equation on the GPU. The implemenation uses the CUDA OpenGL interoperability for visualization.
Currently some optimizations are missing, e.g. the use of shared memory to avoid 
expensive reloading from global GPU memory.


![Test Image 1](data/example.png)

# Installation

Install the dependecies

    glm       0.9.9.8-2     
    glfw-x11  3.3.9-1       
    cmake     3.28.1
    
Dependencies are managed with conan, which can be installed by the python package manager.
For building this project it is necessary to define the proper CUDA paths and architecture in the `CMakeLists.txt` and build from within an `build` folder with:

    cmake .. 
    cmake --build . --config Release

Execution has to be done from the `build` folder in order to link the GLSL shaders properly.

