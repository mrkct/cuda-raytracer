#include <raytracer/HittableList.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Dielectric.h>
#include <raytracer/material/Lambertian.h>
#include <raytracer/material/Metal.h>

__device__ void create_single_sphere_scene(DeviceRNG& rng, HittableList& objects)
{
    auto* material_ground = new Lambertian(rng, Color(0.8, 0.8, 0.0));
    auto* material_center = new Lambertian(rng, Color(0.1, 0.2, 0.5));
    auto* material_left = new Dielectric(rng, 1.5);
    auto* material_right = new Metal(rng, Color(0.8, 0.6, 0.2), 0.0);

    objects.reserve(4);
    objects.append(new Sphere({ 0, 0, -1 }, 0.5, *material_center));
    objects.append(new Sphere({ 1, 0, -1 }, 0.5, *material_right));
    objects.append(new Sphere({ -1, 0, -1 }, -0.4, *material_left));

    objects.append(new Sphere({ 0, -100.5, -1 }, 100, *material_ground));
}