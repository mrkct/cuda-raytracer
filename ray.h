#ifndef RAY_H
#define RAY_H

#include <camera.h>
#include <util/vec3.h>

struct Ray {
    point3 origin;
    vec3 direction;
};
inline __device__ struct Ray make_ray(point3 origin, vec3 direction)
{
    return (struct Ray) { .origin = origin, .direction = direction };
}

inline __device__ point3 ray_at(struct Ray r, float t) { return r.origin + t * r.direction; }

#endif