#ifndef CAMERA_H
#define CAMERA_H

#include <ray.h>
#include <util/vec3.h>

struct Camera {
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

struct Camera make_camera(
    point3 look_from, point3 look_at, vec3 view_up, float vertical_fov, size_t image_width, size_t image_height);

__device__ struct Ray project_ray_from_camera_to_focal_plane(struct Camera, double u, double v);

#endif