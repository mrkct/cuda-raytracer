#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <raytracer/Raytracer.h>
#include <raytracer/util/Vec3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void calculateRay(uint32_t* framebuffer, int image_width, int image_height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= image_height || col >= image_width)
        return;

    uint32_t* pixel = &framebuffer[(image_height - row - 1) * image_width + col];
    *pixel = Vec3(float(col) / image_width, float(row) / image_height, 0.2f).make_rgba();
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