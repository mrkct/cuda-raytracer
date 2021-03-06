#ifndef TEST_SCENE_H
#define TEST_SCENE_H

#include <scenes/scene.h>

#define TEST_SCENE_CAMERA_FROM make_vec3(0, 0, -2)
#define TEST_SCENE_CAMERA_LOOK_AT make_vec3(0, 0, 1)
#define TEST_SCENE_CAMERA_FOV 30

/**
 * @brief Creates a scene with 3 spheres of different materials
 *
 * @return struct Scene
 */
struct Scene create_test_scene(void);

#endif