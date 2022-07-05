#include <scenes/scene.h>

__device__ bool ray_scene_hit(struct Scene scene, struct Ray ray, float t_min, float t_max, HitRecord* out_rec)
{
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (unsigned i = 0; i < scene.spheres_length; i++) {
        Sphere* object = &scene.spheres[i];
        if (ray_sphere_hit(*object, ray, t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *out_rec = temp_rec;
        }
    }

    return hit_anything;
}