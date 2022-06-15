#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using std::sqrt;

class Vec3 {
public:
    __device__ Vec3()
        : e { 0, 0, 0 }
    {
    }
    __device__ Vec3(double e0, double e1, double e2)
        : e { e0, e1, e2 }
    {
    }

    __device__ double x() const { return e[0]; }
    __device__ double y() const { return e[1]; }
    __device__ double z() const { return e[2]; }

    __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __device__ double operator[](int i) const { return e[i]; }
    __device__ double& operator[](int i) { return e[i]; }

    __device__ Vec3& operator+=(Vec3 const& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __device__ Vec3& operator*=(double const t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ Vec3& operator/=(double const t)
    {
        return *this *= 1 / t;
    }

    __device__ double length() const
    {
        return sqrt(length_squared());
    }

    __device__ double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ uint32_t make_rgba() const
    {
        return static_cast<uint8_t>(255) << 24 | static_cast<uint8_t>(255.99f * e[2]) << 16 | static_cast<uint8_t>(255.99f * e[1]) << 8 | static_cast<uint8_t>(255.99f * e[0]);
    }

public:
    double e[3];
};

// Type aliases for Vec3
using Point3 = Vec3; // 3D point
using Color = Vec3;  // RGB color

// Vec3 Utility Functions

inline std::ostream& operator<<(std::ostream& out, Vec3 const& v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

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

__device__ inline Vec3 operator*(double t, Vec3 const& v)
{
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline Vec3 operator*(Vec3 const& v, double t)
{
    return t * v;
}

__device__ inline Vec3 operator/(Vec3 v, double t)
{
    return (1 / t) * v;
}

__device__ inline double dot(Vec3 const& u, Vec3 const& v)
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