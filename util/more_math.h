#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <math.h>

inline __device__ __host__ double degrees_to_radians(double deg)
{
    return deg * M_PI / 180.0;
}

#endif