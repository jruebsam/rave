int main()
{
    dim3 threads(nthread, nthread);
    dim3 grids(width / nthread, height / nthread);

    float dt = 0.00000001;

    solver<<<grids, threads>>>(
        state.T.device, state.T.buffer,
        state.vx.device, state.vx.buffer,
        state.vz.device, state.vz.buffer,
        state.rho.device, state.rho.buffer, width, height, dt);
}

#define LAPLACE(v) (
      inv_dx_quad * (v[East] - 2 * v[Point] + v[West]) 
    + inv_dy_quad * (v[North] - 2 * v[Point] + v[South])
)

__global__ void kernel(float *r_in, float *r_out, int nx, int ny, float dt)
      {
          const unsigned int IDx = blockIdx.x * blockDim.x + threadIdx.x;
          const unsigned int IDy = blockIdx.y * blockDim.y + threadIdx.y;

          int Point = IDy * nx + IDx;
          int North = Point + nx;
          int South = Point - nx;
          int East = Point + 1;
          int West = Point - 1g

                     float dx = 1. / (nx - 1.),
              dy = 1. / (ny - 1.);
          float inv_dx_quad = 1. / (dx * dx);
          float inv_dy_quad = 1. / (dy * dy);

          r_out[Point] = r_in + dt * LAPLACE(r_in);
      }