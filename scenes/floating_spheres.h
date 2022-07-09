#ifndef FLOATING_SPHERES_H
#define FLOATING_SPHERES_H

#include <geometry/sphere.h>
#include <scenes/scene.h>

#define FLOATING_SPHERES_CAMERA_FROM make_vec3(-40, 0, 0)
#define FLOATING_SPHERES_CAMERA_LOOKAT make_vec3(0, 0, 0)
#define FLOATING_SPHERES_CAMERA_FOV 30

/**
 * @brief Create a scene with floating spheres of different materials
 *
 * @return struct Scene
 */
struct Scene create_floating_spheres_scene(void);

#endif