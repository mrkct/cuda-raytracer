#ifndef TEST_SCENE_H
#define TEST_SCENE_H

#include <scenes/scene.h>

#define TEST_SCENE_CAMERA_FROM make_vec3(0, 0, 0)
#define TEST_SCENE_CAMERA_LOOK_AT make_vec3(1, 0, 0)
#define TEST_SCENE_CAMERA_FOV 30

struct Scene create_test_scene(void);

#endif