#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <raytracer/Raytracer.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceArray.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ Color ray_color(Ray const& ray, HittableList const& objects)
{
    HitRecord r;

    if (objects.hit(ray, 0, INFINITY, r)) {
        return 0.5 * (r.normal + Color(1, 1, 1));
    }

    Vec3 unit_direction = unit_vector(ray.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

__global__ void calculate_ray(
    uint32_t* framebuffer,
    HittableList& scene,
    int image_width, int image_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= image_height || col >= image_width)
        return;

    uint32_t* pixel = &framebuffer[(image_height - row - 1) * image_width + col];

    auto const aspect_ratio = (float)image_width / image_height;
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    // FIXME: The origin should go into constant memory
    auto origin = Point3(0, 0, 0);
    auto horizontal = Vec3(viewport_width, 0, 0);
    auto vertical = Vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, focal_length);

    auto u = double(col) / (image_width - 1);
    auto v = double(row) / (image_height - 1);
    Ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);

    *pixel = ray_color(r, scene).make_rgba();
}

__global__ void create_scene(HittableList* list)
{
    new (list) HittableList;
    create_single_sphere_scene(*list);
}

TracedScene Raytracer::trace_scene()
{
    // FIXME: With 32 it starts failing due to 'too many resources requested'
    constexpr int blockSize = 8;

    uint32_t* framebuffer;
    checkCudaErrors(cudaMallocManaged(&framebuffer, m_image.width * m_image.height * 4));
    checkCudaErrors(cudaGetLastError());

    HittableList* device_scene;
    checkCudaErrors(cudaMalloc(&device_scene, sizeof(*device_scene)));
    checkCudaErrors(cudaGetLastError());

    create_scene<<<1, 1>>>(device_scene);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    dim3 grid { (m_image.width + blockSize - 1) / blockSize, (m_image.height + blockSize - 1) / blockSize };
    dim3 blocks { blockSize, blockSize };
    calculate_ray<<<grid, blocks>>>(
        framebuffer,
        *device_scene,
        m_image.width, m_image.height);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return TracedScene { m_image.width, m_image.height, framebuffer };
}