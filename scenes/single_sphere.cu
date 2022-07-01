#include <materials/lambertian.h>
#include <scenes/single_sphere.h>

struct Scene create_single_sphere_scene(void)
{
    struct Scene scene;

    Sphere* sphere;
    Material* material;
    LambertianData* data;

    cudaMallocManaged(&material, sizeof(*material));
    cudaMallocManaged(&data, sizeof(*data));
    cudaMallocManaged(&sphere, sizeof(*sphere));

    *sphere = make_sphere(make_vec3(0, 0, 2), 1, material);
    *data = make_lambertian_material_data(make_color(1, 0, 0));
    *material = make_lambertian_material(data);

    scene.spheres = sphere;
    scene.spheres_length = 1;

    return scene;
}