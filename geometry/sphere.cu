#include <assert.h>
#include <geometry/sphere.h>
#include <math.h>
#include <stdio.h>

__device__ bool ray_sphere_hit(struct Sphere sphere, struct Ray ray, double t_min, double t_max, HitRecord* out_rec)
{
    const vec3 oc = ray.origin - sphere.origin;
    double const a = vec3_length_squared(ray.direction);
    double const half_b = vec3_dot(oc, ray.direction);
    double const c = vec3_length_squared(oc) - sphere.radius * sphere.radius;

    double const discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    double const sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range
    double root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    out_rec->t = root;
    out_rec->p = ray_at(ray, out_rec->t);
    out_rec->material = sphere.material;
    const vec3 outward_normal = (out_rec->p - sphere.origin) / sphere.radius;
    set_face_normal(out_rec, ray, outward_normal);

    double const theta = acos(-outward_normal.y);
    double const phi = atan2(-outward_normal.z, outward_normal.x) + M_PI;
    out_rec->u = phi / (2 * M_PI);
    out_rec->v = theta / M_PI;

    return true;
}
