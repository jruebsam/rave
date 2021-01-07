#pragma once

#include <algorithm>
#include "cuda.h"

struct DataField { 
    float* host;
    float* device;
    float* buffer;

    int nx, ny;
    int* refCount;

    DataField();
    DataField(int nx_, int ny_);

    DataField(const DataField& other);
    DataField& operator=(DataField other);

    ~DataField();

    void toDevice();

    friend void swap(DataField& obj1, DataField& obj2) 
    { 
        using std::swap;
        swap(obj1.host, obj2.host); 
        swap(obj1.device, obj2.device); 
        swap(obj1.buffer, obj2.buffer); 
        swap(obj1.refCount, obj2.refCount); 
        swap(obj1.nx, obj2.nx); 
        swap(obj1.ny, obj2.ny); 
    }

};

struct State {
    DataField T;
    int nx, ny;
    State(){};
    State(int nx_, int ny_): nx{nx_}, ny{ny_}, T(nx_, ny_){};
    void toDevice(){T.toDevice();}
};
