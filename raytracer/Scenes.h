#pragma once

#include <raytracer/HittableList.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/util/DeviceRNG.h>

__device__ void create_single_sphere_scene(DeviceRNG&, HittableList&);