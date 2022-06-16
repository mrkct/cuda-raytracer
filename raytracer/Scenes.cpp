#include <raytracer/HittableList.h>
#include <raytracer/Scenes.h>
#include <raytracer/geometry/Sphere.h>

__device__ void create_single_sphere_scene(HittableList& objects)
{
    objects.reserve(1);
    objects.append(new Sphere({ 0, 0, -1 }, 0.5));
}