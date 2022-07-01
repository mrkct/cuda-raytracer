#include <materials/lambertian.h>
#include <raytracer.h>
#include <stdio.h>
#include <util/check_cuda_errors.h>
#include <util/print_utils.h>
#include <util/rng.h>

static __device__ color ray_color(unsigned id, curandState_t* rng_state, struct Ray ray, struct Scene scene)
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
            vec3 unit_direction = unit_vector(next_ray.direction);
            double t = 0.5 * (unit_direction.y + 1.0);
            return total_attenuation * ((1.0 - t) * make_color(1.0, 1.0, 1.0) + t * make_color(0.5, 0.7, 1.0));
        }
    }

    return make_color(0, 0, 0);
}

static __global__ void calculate_ray(
    struct Framebuffer fb, int samples_per_pixel, struct Scene scene, struct Camera camera)
{
    static unsigned const seed = 1234;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= fb.height || col >= fb.width)
        return;

    unsigned id = (fb.height - row - 1) * fb.width + col;
    uint32_t* pixel = &fb.data[id];

    curandState_t rng_state;
    curand_init(seed + id, 0, 0, &rng_state);

    color pixel_color = make_color(0, 0, 0);
    // FIXME: Questo loop si può fare in parallelo
    for (int s = 0; s < samples_per_pixel; ++s) {
        double u = ((double)col + rng_next(&rng_state)) / (fb.width - 1);
        double v = ((double)row + rng_next(&rng_state)) / (fb.height - 1);
        struct Ray ray = project_ray_from_camera_to_focal_plane(camera, u, v);

        pixel_color = pixel_color + ray_color(id, &rng_state, ray, scene);
        __syncthreads();
    }
    __syncthreads();

    pixel_color = pixel_color / samples_per_pixel;
    pixel_color = gamma2_correct_color(pixel_color);
    *pixel = color_to_rgba(pixel_color);
}

static unsigned const BLOCK_WIDTH = 8;
static unsigned const BLOCK_HEIGHT = 8;

void raytrace_scene(struct Framebuffer fb, struct Scene scene, point3 look_from, point3 look_at, double vfov)
{
    struct Camera camera = make_camera(look_from, look_at, make_vec3(0, 1, 0), vfov, fb.width, fb.height);

    dim3 grid = { (fb.width + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (fb.height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT };
    dim3 block = { BLOCK_WIDTH, BLOCK_HEIGHT };

    calculate_ray<<<grid, block>>>(fb, 100, scene, camera);
    checkCudaErrors(cudaDeviceSynchronize());
}
