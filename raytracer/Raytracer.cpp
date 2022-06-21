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

static constexpr int samples_per_pixel = 100;

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
            if (!rec.material->scatter(id, r, rec, att, scattered_ray)) {
                return { 0.0, 0.0, 0.0 };
            }
            attenuation = attenuation * att;
            r = scattered_ray;
        } else {
            Vec3 unit_direction = unit_vector(ray.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            return attenuation * ((1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0));
        }
    }

    return { 0, 0, 0 };
}

__global__ void calculate_ray(
    uint32_t* framebuffer,
    size_t image_width, size_t image_height,
    int samples_per_pixel,
    DeviceRNG& rng,
    Camera& camera,
    Hittable const& world)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= image_height || col >= image_width)
        return;

    size_t id = (image_height - row - 1) * image_width + col;
    uint32_t* pixel = &framebuffer[id];

    Color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (col + rng.next(id)) / (image_width - 1);
        auto v = (row + rng.next(id)) / (image_height - 1);
        Ray r = camera.get_ray(u, v);
        pixel_color += ray_color(id, r, rng, world);
    }

    pixel_color = pixel_color / samples_per_pixel;
    pixel_color = pixel_color.gamma2_correct();
    *pixel = pixel_color.make_rgba();
}

void Raytracer::trace_scene(DeviceCanvas& canvas, Point3 camera_pos, Point3 look_at, Hittable& world)
{
    auto& camera = Camera::create_on_device(camera_pos, look_at, 60, m_image.width, m_image.height);

    calculate_ray<<<m_grid, m_blocks>>>(
        canvas.pixel_data(),
        canvas.width(), canvas.height(),
        samples_per_pixel,
        *m_rng,
        camera,
        world);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}