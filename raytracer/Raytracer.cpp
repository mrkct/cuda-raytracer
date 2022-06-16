#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <raytracer/Raytracer.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/util/CudaHelper.h>
#include <raytracer/util/Ray.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ double hit_sphere(Point3 const& center, double radius, Ray const& r)
{
    Vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0)
        return -1.0;
    else
        return (-half_b - sqrt(discriminant)) / a;
}

__device__ Color color(Ray const& ray)
{
    auto sphere = Sphere({ 0, 0, -1 }, 0.5);
    HitRecord r;
    if (sphere.hit(ray, 0, 100000000.0, r) && false) {
        return 0.5 * Color(r.normal.x() + 1, r.normal.y() + 1, r.normal.z() + 1);
    } else {
        Vec3 unit_direction = unit_vector(ray.direction());
        auto t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
    }
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

    *pixel = color(r).make_rgba();
}

TracedScene Raytracer::trace_scene(Scene const&)
{
    // FIXME: With 32 it starts failing due to 'too many resources requested'
    constexpr int blockSize = 8;

    uint32_t* framebuffer;
    checkCudaErrors(cudaMallocManaged(&framebuffer, m_image.width * m_image.height * 4));

    dim3 grid { (m_image.width + blockSize - 1) / blockSize, (m_image.height + blockSize - 1) / blockSize };
    dim3 blocks { blockSize, blockSize };
    calculateRay<<<grid, blocks>>>(framebuffer, m_image.width, m_image.height);
    checkCudaErrors(cudaGetLastError());

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    return TracedScene { m_image.width, m_image.height, framebuffer };
}