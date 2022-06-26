#pragma once

#include <iostream>
#include <utility>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, char const* const file, int const line);

template<typename T, typename... Args>
__global__ void __new_on_device_kernel(T* t, std::remove_reference<Args>... args)
{
    // FIXME: Rimuovi questa cosa, troppo facile rompere tutto con questa funzione
    // perchè gli arguments sono passati sempre come const& e non puoi accedervi
    new (t) T(args...);
}

template<typename T>
__global__ void __delete_on_device_kernel(T* t) { delete t; }

template<typename T, typename... Args>
T& new_on_device(std::remove_reference<Args>... args)
{
    T* t;
    checkCudaErrors(cudaMalloc(&t, sizeof(*t)));
    __new_on_device_kernel<T><<<1, 1>>>(t, args...);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return *t;
}

template<typename T>
void delete_on_device(T* p) { __delete_on_device_kernel<T><<<1, 1>>>(p); }