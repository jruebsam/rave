#include "State.h"

State::State(){}

State::State(int nx_, int ny_)
:nx(nx_), ny(ny_)
{
    host.T = xt::zeros<float>({nx, ny});

    cudaMalloc(&(current.T), nx*ny*sizeof(float)); 
    cudaMalloc(&(target.T), nx*ny*sizeof(float)); 
}

void State::toDevice(){
    cudaMemcpy(&(host.T), current.T, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
}

void State::toHost(){
    cudaMemcpy(current.T, &(host.T), nx*ny*sizeof(float), cudaMemcpyHostToDevice);
}

State::~State()
{
    cudaFree(current.T);
    cudaFree(target.T);
}