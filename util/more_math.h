#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <math.h>

inline __device__ __host__ float degrees_to_radians(float deg)
{
    return deg * M_PI / 180.0;
}

/**
 * @brief Returns the minimum value between 2 floating point numbers.
 * This is used because the standard headers only include a version
 * for double precision numbers
 *
 * @param a First value
 * @param b Second value
 * @return The smallest value between a and b
 */
inline __device__ __host__ float my_fmin(float a, float b)
{
    return a < b ? a : b;
}

#endif