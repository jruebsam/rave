#include "Simulation.h"
#include <stdint.h>

__global__ void solver(float* input, float* output, int nx, int ny, float dt)
{
    const unsigned int IDx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;

    float * temp = input;

    int i = IDx*ny + IDy;
    int north = i + nx;
    int south = i - nx;
    int east = i + 1;
    int west = i - 1;

    float dtdx2 =  dt/(0.01*0.01);


    if (IDx > 0 &&  IDx < 1023 && IDy > 0 && IDy < 1023){
        output[i] = temp[i] + dtdx2*((temp[north] -2*temp[i] + temp[south])+ (temp[east] - 2*temp[i] + temp[west]));
    }
}

__global__ void kernel(cudaSurfaceObject_t surface, float* input, double time, double width, double height)
{

    const unsigned int IDx = blockIdx.x * blockDim.x +  threadIdx.x;
    const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;

    float x = IDx/width;
    float y = IDy/height;

    float v = input[IDx + IDy*(int) width];
    //float v = cos(10*x)*sin(10*y)*cos(time)*0.5 + xval + 0.5;

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


    dim3 threads(nthread, nthread);
    dim3 grids(width/nthread, height/nthread);

    solver<<<grids, threads>>>(state.T.device, state.T.buffer, width, height, 0.00001);
    solver<<<grids, threads>>>(state.T.buffer, state.T.device, width, height, 0.00001);
    kernel<<<grids, threads>>>(surface, state.T.device, counter, width, height);

    cudaGraphicsUnmapResources(1, &texRes);
    cudaDestroySurfaceObject(surface);
    counter += 0.01;
}

Simulation::~Simulation()
{
    cudaGraphicsUnregisterResource(texRes);
}
