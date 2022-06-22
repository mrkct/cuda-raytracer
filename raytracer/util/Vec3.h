#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using std::sqrt;

class Vec3 {
public:
    __host__ __device__ Vec3()
        : e { 0, 0, 0 }
    {
    }
    __host__ __device__ Vec3(float e0, float e1, float e2)
        : e { e0, e1, e2 }
    {
    }

    __device__ float x() const { return e[0]; }
    __device__ float y() const { return e[1]; }
    __device__ float z() const { return e[2]; }

    __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __device__ float operator[](int i) const { return e[i]; }
    __device__ float& operator[](int i) { return e[i]; }

    __device__ Vec3& operator+=(Vec3 const& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __device__ Vec3& operator*=(float const t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ Vec3& operator/=(float const t)
    {
        return *this *= 1 / t;
    }

    __device__ float length() const
    {
        return sqrt(length_squared());
    }

    __device__ float length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ uint32_t make_rgba() const
    {
        return static_cast<uint8_t>(255) << 24 | static_cast<uint8_t>(255.99f * e[2]) << 16 | static_cast<uint8_t>(255.99f * e[1]) << 8 | static_cast<uint8_t>(255.99f * e[0]);
    }

    __device__ Vec3 gamma2_correct()
    {
        return { sqrtf(e[0]), sqrtf(e[1]), sqrtf(e[2]) };
    }

    __device__ bool is_near_zero()
    {
        auto const s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

public:
    float e[3];
};

// Type aliases for Vec3
using Point3 = Vec3; // 3D point
using Color = Vec3;  // RGB color

// Vec3 Utility Functions

__device__ inline Vec3 operator+(Vec3 const& u, Vec3 const& v)
{
    return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ inline Vec3 operator-(Vec3 const& u, Vec3 const& v)
{
    return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ inline Vec3 operator*(Vec3 const& u, Vec3 const& v)
{
    return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ inline Vec3 operator*(float t, Vec3 const& v)
{
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline Vec3 operator*(Vec3 const& v, float t)
{
    return t * v;
}

__device__ inline Vec3 operator/(Vec3 v, float t)
{
    return (1 / t) * v;
}

__device__ inline float dot(Vec3 const& u, Vec3 const& v)
{
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__device__ inline Vec3 cross(Vec3 const& u, Vec3 const& v)
{
    return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ inline Vec3 unit_vector(Vec3 v)
{
    return v / v.length();
}

__device__ inline Vec3 reflect_vector(Vec3 vec, Vec3 normal)
{
    // dot(vec, normal) = scale to multiply the normal because 'vec' is not 'unit length'
    // but 'normal' is and we want the reflected vector to be the same length as the original one
    return vec - 2 * dot(vec, normal) * normal;
}