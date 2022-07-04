#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <camera.h>
#include <scenes/scene.h>
#include <util/framebuffer.h>
#include <util/vec3.h>

void raytrace_scene(struct Framebuffer, struct Scene, int samples, point3 look_from, point3 look_at, double vfov);

#endif