#include "Simulation.h"
#include <stdint.h>

__global__ void kernel(cudaSurfaceObject_t surface, double time, double width, double height)
{

    const unsigned int IDx = blockIdx.x * blockDim.x +  threadIdx.x;
    const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;

    float x = IDx/width;
    float y = IDy/height;

    float v = cos(10*x)*sin(10*y)*cos(time)*0.5 + 0.5;
    uint8_t r, g, b; 

    float a=(1-v)/0.25;	
    int x0 = floor(a);
    int y0 = floor(255*(a - x0));

    switch(x0)
    {
        case 0: r=255;    g=y0;     b=0;   break;
        case 1: r=255-y0; g=255;    b=0;   break;
        case 2: r=0;      g=255;    b=y0;  break;
        case 3: r=0;      g=255-y0; b=255; break;
        case 4: r=0;      g=0;      b=255; break;
    }

    uchar4 data = make_uchar4(r, g, b, 0xff);
    surf2Dwrite(data, surface, IDx*sizeof(uchar4), IDy);
}

Simulation::Simulation(){}

Simulation::Simulation(GLuint &texId_handle, int width_, int height_)
: width(width_), height(height_)
{
    cudaGraphicsGLRegisterImage(&texRes, texId_handle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    state = State(width, height);
}


void Simulation::Step()
{
    cudaGraphicsMapResources(1, &texRes);
    cudaGraphicsSubResourceGetMappedArray(&data, texRes, 0, 0);

    cudaResourceDesc resoureDescription;
    cudaSurfaceObject_t surface = 0;

    memset(&resoureDescription, 0, sizeof(resoureDescription));
    resoureDescription.resType = cudaResourceTypeArray;   
    resoureDescription.res.array.array = data; 

    cudaCreateSurfaceObject(&surface, &resoureDescription);

    int nthread = 32;


    dim3 grids(nthread, nthread);
    dim3 threads(width/nthread, height/nthread);
    kernel<<<grids, threads>>>(surface, counter, width, height);

    cudaGraphicsUnmapResources(1, &texRes);
    cudaDestroySurfaceObject(surface);
    counter += 0.01;
}

Simulation::~Simulation()
{
    cudaGraphicsUnregisterResource(texRes);
}
