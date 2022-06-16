#pragma once

#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        std::cerr << "Message: '" << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}
