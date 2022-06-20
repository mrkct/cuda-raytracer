#pragma once

#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line);

template<typename T, typename... Args>
__global__ void __new_on_device_kernel(T* t, Args&&... args)
{
    new (t) T(args...);
}

template<typename T, typename... Args>
T& new_on_device(Args&&... args)
{
    T* t;
    checkCudaErrors(cudaMalloc(&t, sizeof(*t)));
    __new_on_device_kernel<T><<<1, 1>>>(t, args...);
    cudaDeviceSynchronize();

    return *t;
}