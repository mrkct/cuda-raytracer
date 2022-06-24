#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>

static __global__ void kinit(
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
}

DeviceRNG::Builder DeviceRNG::init(dim3 grid, dim3 block, size_t image_width, size_t image_height)
{
    curandState* rng_state;
    checkCudaErrors(cudaMalloc(&rng_state, image_width * image_height * sizeof(*rng_state)));

    kinit<<<grid, block>>>(
        rng_state,
        5723, // FIXME: Change to a random seed
        image_width,
        image_height);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    return Builder { .state = rng_state };
}

__device__ float DeviceRNG::next()
{
    // curand_uniform returns between the range (0.0, 1] but we
    // actually want the range [0.0, 1.0)
    return 1.0 - curand_uniform(m_curand_state);
}

__device__ float DeviceRNG::next(float min, float max)
{
    return min + next() * (max - min);
}

__device__ Vec3 DeviceRNG::next_in_unit_sphere()
{
    // See: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    auto u = next();
    auto v = next();
    auto theta = u * 2.0 * M_PI;
    auto phi = acosf(2.0 * v - 1.0);
    auto r = cbrtf(next());
    auto sin_theta = sinf(theta);
    auto cos_theta = cosf(theta);
    auto sin_phi = sinf(phi);
    auto cos_phi = cosf(phi);
    auto x = r * sin_phi * cos_theta;
    auto y = r * sin_phi * sin_theta;
    auto z = r * cos_phi;
    return { x, y, z };
}