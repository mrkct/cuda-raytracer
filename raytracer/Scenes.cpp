#include <raytracer/HittableList.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Lambertian.h>
#include <raytracer/material/Metal.h>

__device__ void create_single_sphere_scene(DeviceRNG& rng, HittableList& objects)
{
    auto* red_lambertian = new Lambertian(rng, Color(1.0, 0.0, 0.0));
    auto* green_lambertian = new Lambertian(rng, Color(0.0, 1.0, 0.0));
    auto* white_fuzzy_metal = new Metal(rng, Color { 0.8, 0.8, 0.8 }, 0.7);
    auto* blue_clear_metal = new Metal(rng, Color { 0.2, 0.2, 1.0 }, 0.1);

    objects.reserve(4);
    objects.append(new Sphere({ 0, 0, -1 }, 0.5, *red_lambertian));
    objects.append(new Sphere({ 1, 0, -1 }, 0.5, *white_fuzzy_metal));
    objects.append(new Sphere({ -1, 0, -1 }, 0.5, *blue_clear_metal));

    objects.append(new Sphere({ 0, -100.5, -1 }, 100, *green_lambertian));
}