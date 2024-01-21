#include "Simulation.h"
#include <stdint.h>

#include "colormap.cu"

#define DERIV_X(v) (1/(2*dx)*(v[East]  - v[West]))
#define DERIV_Z(v) (1/(2*dz)*(v[North] - v[South]))

#define LAPLACE(v) (inv_dx_quad*(v[East]  -2*v[Point] + v[West] )\
                   +inv_dz_quad*(v[North] -2*v[Point] + v[South]))

#define ADVECT(v)((axp*(v[Point] - v[West])/dx  + axm*(v[East]  - v[Point])/dx \
                 + azp*(v[Point] - v[South])/dz + azm*(v[North] - v[Point])/dz))


__global__ void solver
(
    float* in_T,
    float* out_T,
    float* in_vx,
    float* out_vx,
    float* in_vz,
    float* out_vz,
    float* in_rho,
    float* out_rho,
    int nx, 
    int ny, 
    float dt)
{
    const unsigned int IDx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;

    int Point = IDy*nx + IDx;
    int North = Point + nx;
    int South = Point - nx;
    int East = Point + 1;
    int West = Point - 1;

    float dx = 3./(nx - 1.),  dz = 1./(ny -1.);
    float c2 = 1000000;
    float Pr = 0.7;
    float Ra = 5000000;
    float inv_dx_quad = 1./(dx*dx);
    float inv_dz_quad = 1./(dz*dz);



    if (IDx > 1 &&  IDx < nx - 1 && IDy > 1 && IDy < ny - 1){
        float axp = max(in_vx[Point], 0.0f);
        float axm = min(in_vx[Point], 0.0f);
        float azp = max(in_vz[Point], 0.0f);
        float azm = min(in_vz[Point], 0.0f);


        out_rho[Point] = in_rho[Point] + dt*(-DERIV_X(in_vx) - DERIV_Z(in_vz));
        out_vx[Point]  = in_vx[Point]  + dt*(-ADVECT(in_vx) - c2*DERIV_X(in_rho) + Pr*LAPLACE(in_vx));
        out_vz[Point]  = in_vz[Point]  + dt*(-ADVECT(in_vz) - c2*DERIV_Z(in_rho) + Pr*LAPLACE(in_vz) + Pr*Ra*in_T[Point]);
        out_T[Point]   = in_T[Point]   + dt*(-ADVECT(in_T) + LAPLACE(in_T) + in_vz[Point]);
    }
}


__global__ void kernel(cudaSurfaceObject_t surface, float* input, int nx, int ny)
{

    const unsigned int IDx = blockIdx.x * blockDim.x +  threadIdx.x;
    const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;


    float y = IDy/(float)ny;
    float v = input[IDx + IDy*nx] + (1- y);

    uint8_t r, g, b; 

    r = (int) (colormap_red(v)*255);
    g = (int) (colormap_green(v)*255);
    b = (int) (colormap_blue(v)*255);

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

    float dt = 0.000001;

    solver<<<grids, threads>>> (
        state.T.device, state.T.buffer,
        state.vx.device, state.vx.buffer,
        state.vz.device, state.vz.buffer,
        state.rho.device, state.rho.buffer, width, height, dt
    );

    solver<<<grids, threads>>> (
        state.T.buffer, state.T.device,
        state.vx.buffer, state.vx.device,
        state.vz.buffer, state.vz.device,
        state.rho.buffer, state.rho.device, width, height, dt
    );
        
    kernel<<<grids, threads>>>(surface, state.T.device, width, height);

    cudaGraphicsUnmapResources(1, &texRes);
    cudaDestroySurfaceObject(surface);
}

Simulation::~Simulation()
{
    cudaGraphicsUnregisterResource(texRes);
}
