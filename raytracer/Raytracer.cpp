#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <raytracer/Camera.h>
#include <raytracer/Raytracer.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceArray.h>
#include <raytracer/util/DeviceRNG.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static constexpr int samples_per_pixel = 100;

__device__ Color ray_color(size_t id, Ray const& ray, DeviceRNG& rng, HittableList const& objects)
{
    HitRecord rec;

    Ray r = ray;
    Color attenuation { 1.0, 1.0, 1.0 };
    static constexpr int max_depth = 50;
    for (int i = 0; i < max_depth; i++) {
        if (objects.hit(r, 0.01f, INFINITY, rec)) {
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
    DeviceRNG& rng,
    HittableList& scene,
    size_t image_width, size_t image_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= image_height || col >= image_width)
        return;

    size_t id = (image_height - row - 1) * image_width + col;
    uint32_t* pixel = &framebuffer[id];
    Camera camera {
        { -2, 2, 1 }, { 0, 0, -1 }, { 0, 1, 0 }, 90,
        image_width, image_height
    };

    Color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (col + rng.next(id)) / (image_width - 1);
        auto v = (row + rng.next(id)) / (image_height - 1);
        Ray r = camera.get_ray(u, v);
        pixel_color += ray_color(id, r, rng, scene);
    }

    pixel_color = pixel_color / samples_per_pixel;
    pixel_color = pixel_color.gamma2_correct();
    *pixel = pixel_color.make_rgba();
}

__global__ void create_scene(DeviceRNG& rng, HittableList* list)
{
    new (list) HittableList;
    create_single_sphere_scene(rng, *list);
}

TracedScene Raytracer::trace_scene()
{
    // FIXME: With 32 it starts failing due to 'too many resources requested'
    constexpr int blockSize = 8;

    uint32_t* framebuffer;
    checkCudaErrors(cudaMallocManaged(&framebuffer, m_image.width * m_image.height * 4));
    checkCudaErrors(cudaGetLastError());

    dim3 grid { (m_image.width + blockSize - 1) / blockSize, (m_image.height + blockSize - 1) / blockSize };
    dim3 blocks { blockSize, blockSize };

    auto& device_rng = *DeviceRNG::init(grid, blocks, m_image.width, m_image.height);

    HittableList* device_scene;
    checkCudaErrors(cudaMalloc(&device_scene, sizeof(*device_scene)));
    checkCudaErrors(cudaGetLastError());

    create_scene<<<1, 1>>>(device_rng, device_scene);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    calculate_ray<<<grid, blocks>>>(
        framebuffer,
        device_rng,
        *device_scene,
        m_image.width, m_image.height);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return TracedScene { m_image.width, m_image.height, framebuffer };
}