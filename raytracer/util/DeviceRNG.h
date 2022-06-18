#pragma once

#include <curand_kernel.h>
#include <raytracer/util/Vec3.h>

class DeviceRNG {
public:
    static DeviceRNG* init(dim3 grid, dim3 block, size_t image_width, size_t image_height);

    __device__ float next(size_t id);
    __device__ float next(size_t id, float min, float max);
    __device__ Vec3 next_in_unit_sphere(size_t id);

    __device__ DeviceRNG(curandState*);

private:
    curandState* m_curand_state;
};