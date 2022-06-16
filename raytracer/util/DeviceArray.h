#pragma once

#include <cstddef>

template<typename T>
struct DeviceArray {
    __device__ T const& at(size_t i) const { return elements[i]; }
    __device__ T const& operator[](size_t i) const { return at(i); }
    __device__ void append(T t) { elements[length++] = t; }

    T* elements;
    size_t length;
};