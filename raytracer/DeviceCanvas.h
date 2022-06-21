#pragma once

#include <cstdint>
#include <memory>
#include <raytracer/util/CudaHelpers.h>

class DeviceCanvas {
public:
    DeviceCanvas(unsigned int width, unsigned int height)
        : m_width(width)
        , m_height(height)
    {
        checkCudaErrors(cudaMallocManaged(&m_data, width * height * 4));
        checkCudaErrors(cudaGetLastError());
    }

    ~DeviceCanvas() { checkCudaErrors(cudaFree(m_data)); }

    int width() const { return m_width; }
    int height() const { return m_height; }
    static int bytes_per_pixel() { return 4; }
    uint32_t* pixel_data() { return m_data; }

    int write_to_file(char const* path);

private:
    unsigned int m_width, m_height;
    uint32_t* m_data;
};
