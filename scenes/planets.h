#ifndef PLANETS_H
#define PLANETS_H

#include <geometry/sphere.h>
#include <scenes/scene.h>

#define PLANETS_CAMERA_FROM make_vec3(-40, 35, -10)
#define PLANETS_CAMERA_LOOKAT make_vec3(-1, 0, 30)
#define PLANETS_CAMERA_FOV 30

/**
 * @brief Create a scene with spheres using textures of
 * planets, simulating the first 5 planets in order from the sun
 *
 * @return struct Scene
 */
struct Scene create_planets_scene(void);

#endif