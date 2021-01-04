#pragma once

#include <string>
#include <map>

#include "xtensor/xarray.hpp"
#include "cuda.h"

struct HostFields  {
    xt::xarray<float> T;
};

struct DataFields { 
    float* T;
};

class State
{
private:
    int nx, ny;
public:
    HostFields host;
    DataFields current;
    DataFields target;

    State();
    State(int nx_, int ny_);
    void toHost();
    void toDevice();
    ~State();
};
