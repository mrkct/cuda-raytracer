#include <camera.h>
#include <math.h>
#include <util/more_math.h>

struct Camera make_camera(
    point3 look_from, point3 look_at, vec3 view_up, float vertical_fov, size_t image_width, size_t image_height)
{
    struct Camera camera;

    double const aspect_ratio = (double)image_width / image_height;

    double theta = degrees_to_radians(vertical_fov);
    double h = tan(theta / 2.0f);
    double viewport_height = 2.0 * h;
    double viewport_width = aspect_ratio * viewport_height;

    vec3 w = unit_vector(look_from - look_at);
    vec3 u = unit_vector(vec3_cross(view_up, w));
    vec3 v = vec3_cross(w, u);

    camera.origin = look_from;
    camera.horizontal = viewport_width * u;
    camera.vertical = viewport_height * v;
    camera.lower_left_corner = camera.origin - camera.horizontal / 2 - camera.vertical / 2 - w;

    return camera;
}

__device__ struct Ray project_ray_from_camera_to_focal_plane(struct Camera camera, double u, double v)
{
    return (struct Ray) { .origin = camera.origin,
        .direction = camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin };
}