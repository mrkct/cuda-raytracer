#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <math.h>

inline __device__ __host__ float degrees_to_radians(float deg)
{
    return deg * M_PI / 180.0;
}

inline __device__ __host__ float my_fmin(float a, float b)
{
    return a < b ? a : b;
}

#endif