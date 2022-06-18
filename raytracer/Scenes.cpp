#include <raytracer/HittableList.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Lambertian.h>

__device__ void create_single_sphere_scene(DeviceRNG& rng, HittableList& objects)
{
    auto* red_lambertian = new Lambertian(rng, Color(1.0, 0.0, 0.0));
    auto* green_lambertian = new Lambertian(rng, Color(0.0, 1.0, 0.0));

    objects.reserve(2);
    objects.append(new Sphere({ 0, 0, -1 }, 0.5, *red_lambertian));
    objects.append(new Sphere({ 0, -100.5, -1 }, 100, *green_lambertian));
}