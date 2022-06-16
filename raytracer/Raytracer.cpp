#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <raytracer/Raytracer.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ bool hit_sphere(Point3 const& center, double radius, Ray const& r)
{
    Vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

__global__ void calculateRay(uint32_t* framebuffer, int image_width, int image_height)
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

    if (hit_sphere(Point3(0.0, 0.0, -1.0), 0.5, r)) {
        *pixel = Color(1.0, 0.0, 0.0).make_rgba();
    } else {
        Vec3 unit_direction = unit_vector(r.direction());
        auto t = 0.5 * (unit_direction.y() + 1.0);
        *pixel = ((1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)).make_rgba();
    }
}

TracedScene Raytracer::trace_scene(Scene const&)
{
    constexpr int blockSize = 32;

    uint32_t* framebuffer;
    cudaMallocManaged(&framebuffer, m_image.width * m_image.height * 4);

    dim3 grid { (m_image.width + blockSize - 1) / blockSize, (m_image.height + blockSize - 1) / blockSize };
    dim3 blocks { blockSize, blockSize };
    calculateRay<<<grid, blocks>>>(framebuffer, m_image.width, m_image.height);
    cudaDeviceSynchronize();

    return TracedScene { m_image.width, m_image.height, framebuffer };
}