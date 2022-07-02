#include <materials/lambertian.h>
#include <scenes/single_sphere.h>
#include <textures/checkered_texture.h>

struct Scene create_single_sphere_scene(void)
{
    struct Scene scene;

    Sphere* sphere;
    Material* material;
    LambertianData* material_data;
    Texture* texture;
    CheckeredTextureData* texture_data;

    cudaMallocManaged(&material, sizeof(*material));
    cudaMallocManaged(&material_data, sizeof(*material_data));
    cudaMallocManaged(&sphere, sizeof(*sphere));
    cudaMallocManaged(&texture, sizeof(*texture));
    cudaMallocManaged(&texture_data, sizeof(*texture_data));

    *texture_data = make_checkered_texture_data(0.1, make_color(0.8, 0, 0), make_color(0.9, 0.9, 0.9));
    *texture = make_checkered_texture(texture_data);
    *material_data = make_lambertian_material_data(texture);
    *material = make_lambertian_material(material_data);
    *sphere = make_sphere(make_vec3(0, 0, 2), 1, material);

    scene.spheres = sphere;
    scene.spheres_length = 1;

    return scene;
}