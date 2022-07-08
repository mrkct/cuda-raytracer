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

/**
 * @brief Creates a camera object with the following properties
 *
 * @param look_from A point in 3d space where the camera should be positioned
 * @param look_at A point in 3d space where the camera should be looking at
 * @param view_up A vector indicating which way is 'up'. This can be used to rotate the camera on itself
 * @param vertical_fov Vertical field of view of the camera, only values between 0 and 180 degrees are supported
 * @param image_width Width in pixel of the final image
 * @param image_height Height in pixels of the final image
 * @return struct Camera
 */
struct Camera make_camera(
    point3 look_from, point3 look_at, vec3 view_up, float vertical_fov, size_t image_width, size_t image_height);

/**
 * @brief Creates a ray with origin in the camera position and direction
 * towards the 'u' 'v' coordinates on the focal plane
 *
 * @param u: A value between 0 and 1 representing the relative horizontal position on the focal plane
 * @param v: A value between 0 and 1 representing the relative vertical position on the focal plane
 * @return struct Ray
 */
__device__ struct Ray project_ray_from_camera_to_focal_plane(struct Camera, float u, float v);

#endif