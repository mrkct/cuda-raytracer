#ifndef PLANETS_H
#define PLANETS_H

#include <geometry/sphere.h>
#include <scenes/scene.h>

#define PLANETS_CAMERA_FROM make_vec3(-40, 35, -10)
#define PLANETS_CAMERA_LOOKAT make_vec3(-1, 0, 30)
#define PLANETS_CAMERA_FOV 30

struct Scene create_planets_scene(void);

#endif