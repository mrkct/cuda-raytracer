#ifndef SINGLE_SPHERE_H
#define SINGLE_SPHERE_H

#include <geometry/sphere.h>
#include <scenes/scene.h>

/**
 * @brief Create a scene with a single sphere of red, opaque material
 *
 * @return struct Scene
 */
struct Scene create_single_sphere_scene(void);

#endif