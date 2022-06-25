#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <raytracer/Camera.h>
#include <raytracer/Raytracer.h>
#include <raytracer/material/Material.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr int samples_per_pixel = 1;

#define tprintf       \
    if (id == 206273) \
    printf

__device__ Color ray_color(
    size_t id,
    Ray const& ray,
    DeviceRNG& rng,
    Hittable const& world)
{
    HitRecord rec;

    Ray r = ray;
    Color attenuation { 1.0, 1.0, 1.0 };
    static constexpr int max_depth = 50;
    for (int i = 0; i < max_depth; i++) {
        if (world.hit(r, 0.01f, INFINITY, rec)) {
            Color att;
            Ray scattered_ray;
            tprintf(
                "id: %u\tHIT\tt=%.6f front=%d p=(%.6f %.6f %.6f)  norm=(%.6f %.6f %.6f)\n",
                id, rec.t, (int)rec.front_face, rec.p.x(), rec.p.y(), rec.p.z(), rec.normal.x(), rec.normal.y(), rec.normal.z());
            if (!rec.material->scatter(rng, r, rec, att, scattered_ray)) {
                return { 0.0, 0.0, 0.0 };
            }
            attenuation = attenuation * att;
            r = scattered_ray;
        } else {
            Vec3 unit_direction = unit_vector(ray.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            tprintf("id: %u\tNO HIT\n", id);
            return attenuation * ((1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0));
        }
    }

    return { 0, 0, 0 };
}

__global__ void calculate_ray(
    uint32_t* framebuffer,
    size_t image_width, size_t image_height,
    int samples_per_pixel,
    DeviceRNG::Builder rng_builder,
    Camera::Builder camera_builder,
    Hittable const& world)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= image_height || col >= image_width)
        return;

    size_t id = (image_height - row - 1) * image_width + col;
    uint32_t* pixel = &framebuffer[id];

    Camera camera = camera_builder.build();
    DeviceRNG rng = rng_builder.build(id);

    Color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = ((double)col /* + rng.next()*/) / (image_width - 1);
        auto v = ((double)row /* + rng.next()*/) / (image_height - 1);
        Ray r = camera.get_ray(u, v);
        tprintf("id: %u\tu=%.6f\tv=%.6f\tRay(origin=%.6f %.6f %.6f   direction=%.6f %.6f %.6f)\n", id, u, v, r.origin().x(), r.origin().y(), r.origin().z(), r.direction().x(), r.direction().y(), r.direction().z());
        pixel_color += ray_color(id, r, rng, world);
        __syncthreads();
    }
    __syncthreads();

    tprintf("id: %u\tcolor: (%.6f %.6f %.6f)\n", id, pixel_color.x(), pixel_color.y(), pixel_color.z());
    pixel_color = pixel_color / samples_per_pixel;
    tprintf("id: %u\tcolor/samples: (%.6f %.6f %.6f)\n", id, pixel_color.x(), pixel_color.y(), pixel_color.z());
    pixel_color = pixel_color.gamma2_correct();
    tprintf("id: %u\tcolor corrected: (%.6f %.6f %.6f)\n", id, pixel_color.x(), pixel_color.y(), pixel_color.z());
    *pixel = pixel_color.make_rgba();
    __syncthreads();
    tprintf("id: %u\trgba: %x\n", id, *pixel);
}

void Raytracer::trace_scene(DeviceCanvas& canvas, Point3 camera_pos, Point3 look_at, Hittable& world)
{
    calculate_ray<<<m_grid, m_blocks>>>(
        canvas.pixel_data(),
        canvas.width(), canvas.height(),
        samples_per_pixel,
        m_rng_builder,
        Camera::Builder { .look_from = camera_pos, .look_at = look_at, .vertical_fov = 60, .image_width = m_image.width, .image_height = m_image.height },
        world);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}