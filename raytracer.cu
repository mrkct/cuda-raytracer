#include <assert.h>
#include <materials/lambertian.h>
#include <raytracer.h>
#include <stdio.h>
#include <util/check_cuda_errors.h>
#include <util/print_utils.h>
#include <util/rng.h>

__constant__ struct Camera camera;
__constant__ struct Scene scene;
__constant__ struct Framebuffer fb;

static __device__ color ray_color(unsigned id, curandState_t* rng_state, struct Ray const& ray)
{
    struct HitRecord rec;
    struct Ray next_ray = ray;
    color total_attenuation = make_color(1.0, 1.0, 1.0);

    static int const max_depth = 50;
    for (int i = 0; i < max_depth; i++) {
        if (ray_scene_hit(scene, next_ray, 0.01f, INFINITY, &rec)) {
            color attenuation;
            Ray scattered_ray;

            if (!material_scatter(rec.material, rng_state, next_ray, &rec, &attenuation, &scattered_ray)) {
                return make_color(0, 0, 0);
            }
            total_attenuation = total_attenuation * attenuation;
            next_ray = scattered_ray;
        } else {
            vec3 const unit_direction = unit_vector(next_ray.direction);
            float const t = 0.5 * (unit_direction.y + 1.0);
            return total_attenuation * ((1.0 - t) * make_color(1.0, 1.0, 1.0) + t * make_color(0.5, 0.7, 1.0));
        }
    }

    return make_color(0, 0, 0);
}

static __device__ color calculate_ray(
    unsigned const id,
    int const row,
    int const col,
    curandState_t* rng_state)
{
    float const u = ((float)col + rng_next(rng_state)) / (fb.width - 1);
    float const v = ((float)row + rng_next(rng_state)) / (fb.height - 1);
    const struct Ray ray = project_ray_from_camera_to_focal_plane(camera, u, v);

    return ray_color(id, rng_state, ray);
}

static __global__ void trace_scene()
{
    static unsigned const seed = 1234;

    int const col = blockIdx.x * blockDim.x + threadIdx.x;
    int const row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= fb.height || col >= fb.width)
        return;

    unsigned const id = (fb.height - row - 1) * fb.width + col;
    color* pixel = &fb.color_data[id];

    curandState_t rng_state;
    curand_init(seed + id * blockIdx.z, 0, 0, &rng_state);

    color pixel_color = calculate_ray(id, row, col, &rng_state);

    __syncthreads();
    atomicAdd(&pixel->x, pixel_color.x);
    atomicAdd(&pixel->y, pixel_color.y);
    atomicAdd(&pixel->z, pixel_color.z);
}

static __global__ void convert_from_vec3_to_rgba(struct Framebuffer fb, int samples_per_pixel)
{
    int const col = blockIdx.x * blockDim.x + threadIdx.x;
    int const row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= fb.height || col >= fb.width)
        return;

    unsigned const id = (fb.height - row - 1) * fb.width + col;

    color pixel_color = fb.color_data[id];

    pixel_color = pixel_color / samples_per_pixel;
    pixel_color = gamma2_correct_color(pixel_color);

    uint32_t* pixel = &fb.data[id];
    *pixel = color_to_rgba(pixel_color);
}

static unsigned const BLOCK_WIDTH = 8;
static unsigned const BLOCK_HEIGHT = 8;

void raytrace_scene(struct Framebuffer framebuffer, struct Scene local_scene, int samples, point3 look_from, point3 look_at, float vfov)
{
    struct Camera local_camera = make_camera(look_from, look_at, make_vec3(0, 1, 0), vfov, framebuffer.width, framebuffer.height);
    checkCudaErrors(cudaMemcpyToSymbol(camera, &local_camera, sizeof(local_camera)));
    checkCudaErrors(cudaMemcpyToSymbol(scene, &local_scene, sizeof(local_scene)));
    checkCudaErrors(cudaMemcpyToSymbol(fb, &framebuffer, sizeof(framebuffer)));
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 grid = {
        (framebuffer.width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
        (framebuffer.height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
        (unsigned)samples
    };
    dim3 block = { BLOCK_WIDTH, BLOCK_HEIGHT };

    trace_scene<<<grid, block>>>();
    checkCudaErrors(cudaDeviceSynchronize());
    grid.z = 1;
    convert_from_vec3_to_rgba<<<grid, block>>>(framebuffer, samples);
    checkCudaErrors(cudaDeviceSynchronize());
}
