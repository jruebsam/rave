#include "State.h"
#include "cuda_helpers.h"

DataField::DataField()
{
    host   = nullptr;
    device = nullptr; 
    buffer = nullptr;
    refCount = nullptr;
    nx=0;
    ny=0;
};

DataField::DataField(int nx_, int ny_) : nx{nx_}, ny{ny_}
{

    host = new float[nx*ny]();
    CUDACHECK(cudaMalloc(&(device), nx*ny*sizeof(float))); 
    CUDACHECK(cudaMalloc(&(buffer), nx*ny*sizeof(float))); 

    refCount = new int(0);
    *refCount += 1;
};

DataField::DataField(const DataField& other)
{
    if(this != &other){
        this->host = other.host;
        this->device = other.device;
        this->buffer = other.buffer;
        this->nx = other.nx;
        this->ny = other.ny;
        this->refCount = other.refCount;
        *refCount += 1;
    }
};

DataField& DataField::operator=(DataField other){
    swap(*this, other); 
    return *this; 
};

DataField::~DataField(){
    if (refCount != nullptr)
    {
        if (*refCount == 1) {
            delete[] this->host;
            delete refCount;

            CUDACHECK(cudaFree(this->device));
            CUDACHECK(cudaFree(this->buffer));
        } else {
            *refCount -= 1;
        }

        this->host = nullptr;
        this->device = nullptr;
        this->buffer = nullptr;
        this->refCount = nullptr;
    }
};

void DataField::toDevice(){
    CUDACHECK(cudaMemcpy(device, host, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
};