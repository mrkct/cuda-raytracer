#ifndef VEC3_H
#define VEC3_H

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

typedef float4 vec3;
typedef vec3 point3;
typedef vec3 color;

inline __host__ __device__ vec3 make_vec3(float x, float y, float z) { return make_float4(x, y, z, 0); }

inline __host__ __device__ point3 make_point3(float x, float y, float z) { return make_vec3(x, y, z); }

inline __host__ __device__ color make_color(float r, float g, float b) { return make_vec3(r, g, b); }

inline __host__ __device__ float vec3_length_squared(vec3 const& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

inline __host__ __device__ float vec3_length(vec3 const& v) { return sqrt(vec3_length_squared(v)); }

inline __host__ __device__ uint32_t color_to_rgba(color const& c)
{
    return ((uint8_t)(255) << 24 | (uint8_t)(255.99f * c.z) << 16 | (uint8_t)(255.99f * c.y) << 8
        | (uint8_t)(255.99f * c.x));
}

inline __host__ __device__ color rgba_to_color(uint32_t rgba)
{
    return make_color((float)(rgba & 0xff) / 255, (float)((rgba >> 8) & 0xff) / 255, (float)((rgba >> 16) & 0xff) / 255);
}

inline __host__ __device__ color gamma2_correct_color(color const& c) { return make_vec3(sqrt(c.x), sqrt(c.y), sqrt(c.z)); }

inline __host__ __device__ bool is_near_zero(vec3 const& v)
{
    float const s = 1e-8;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

inline __host__ __device__ vec3 vec3_negate(vec3 const& v) { return make_vec3(-v.x, -v.y, -v.z); }

inline __host__ __device__ vec3 operator+(vec3 const& u, vec3 const& v)
{
    return make_vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

inline __host__ __device__ vec3 operator-(vec3 const& u, vec3 const& v)
{
    return make_vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

inline __host__ __device__ vec3 operator*(vec3 const& u, vec3 const& v)
{
    return make_vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline __host__ __device__ vec3 operator*(float t, vec3 const& v) { return make_vec3(t * v.x, t * v.y, t * v.z); }

inline __host__ __device__ vec3 operator*(vec3 const& v, float t) { return t * v; }

inline __host__ __device__ vec3 operator/(vec3 const& v, float t) { return (1.0 / t) * v; }

inline __host__ __device__ float vec3_dot(vec3 const& u, vec3 const& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }

inline __host__ __device__ vec3 vec3_cross(vec3 const& u, vec3 const& v)
{
    return make_vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

inline __host__ __device__ vec3 unit_vector(vec3 const& v) { return v / vec3_length(v); }

inline __host__ __device__ vec3 reflect_vector(vec3 const& vec, vec3 const& normal)
{
    // dot(vec, normal) = scale to multiply the normal because 'vec' is not 'unit length'
    // but 'normal' is and we want the reflected vector to be the same length as the original one
    return vec - 2 * vec3_dot(vec, normal) * normal;
}

#endif