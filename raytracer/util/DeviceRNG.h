#pragma once

#include <curand_kernel.h>
#include <raytracer/util/Vec3.h>

class DeviceRNG {
public:
    struct Builder {
        curandState* state;

        __device__ DeviceRNG build(size_t id) { return DeviceRNG(&state[id]); }
    };

    static Builder init(dim3 grid, dim3 block, size_t image_width, size_t image_height);

    __device__ float next();
    __device__ float next(float min, float max);
    __device__ Vec3 next_in_unit_sphere();

    __device__ DeviceRNG(curandState* state)
        : m_curand_state(state)
    {
    }

private:
    curandState* m_curand_state;
};