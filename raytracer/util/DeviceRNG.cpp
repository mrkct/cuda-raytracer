#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>

__device__ DeviceRNG::DeviceRNG(curandState* curand_state)
    : m_curand_state(curand_state)
{
}

static __global__ void kinit(
    DeviceRNG* rng,
    curandState* state_buffer,
    unsigned long long seed,
    size_t width, size_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    int pixel_offset = (height - row - 1) * width + col;
    curand_init(seed, pixel_offset, 0, &state_buffer[pixel_offset]);

    if (col == 0 && row == 0)
        new (rng) DeviceRNG(state_buffer);
}

DeviceRNG* DeviceRNG::init(dim3 grid, dim3 block, size_t image_width, size_t image_height)
{
    DeviceRNG* rng;
    checkCudaErrors(cudaMalloc(&rng, sizeof(DeviceRNG)));

    curandState* rng_state;
    checkCudaErrors(cudaMalloc(&rng_state, image_width * image_height * sizeof(*rng_state)));

    kinit<<<grid, block>>>(
        rng,
        rng_state,
        5723,
        image_width,
        image_height);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    return rng;
}

__device__ float DeviceRNG::next(size_t id)
{
    // curand_uniform returns between the range (0.0, 1] but we
    // actually want the range [0.0, 1.0)
    return 1.0 - curand_uniform(&m_curand_state[id]);
}

__device__ float DeviceRNG::next(size_t id, float min, float max)
{
    return min + next(id) * (max - min);
}