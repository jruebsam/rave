#pragma once

#include "glad/glad.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "State.h"


class Simulation
{
private:
    double counter = 0.0;

    GLuint texId;
    cudaGraphicsResource_t texRes;
    cudaArray_t data;
    int width, height;

public:
    Simulation();
    Simulation(GLuint& texId_handle, int width_, int height_);
    void Step();
    State state;
    ~Simulation();
};

