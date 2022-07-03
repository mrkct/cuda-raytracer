#include <materials/lambertian.h>
#include <materials/metal.h>
#include <scenes/planets.h>
#include <textures/image_texture.h>
#include <util/check_cuda_errors.h>

static int const N_SPHERES = 6;

struct Scene create_planets_scene(void)
{
    struct Scene scene;

    Sphere* spheres;
    checkCudaErrors(cudaMallocManaged(&spheres, sizeof(Sphere) * N_SPHERES));

    Material* materials;
    checkCudaErrors(cudaMallocManaged(&materials, sizeof(Material) * N_SPHERES));

    LambertianData* planets_material_data;
    checkCudaErrors(cudaMallocManaged(&planets_material_data, sizeof(*planets_material_data) * (N_SPHERES - 1)));

    Texture* planets_textures;
    checkCudaErrors(cudaMallocManaged(&planets_textures, sizeof(Texture) * (N_SPHERES - 1)));

    ImageTextureData* planets_textures_data;
    checkCudaErrors(cudaMallocManaged(&planets_textures_data, sizeof(ImageTextureData) * (N_SPHERES - 1)));

    planets_textures_data[0] = make_image_texture_data_from_file("sample_textures/mercury.png");
    planets_textures_data[1] = make_image_texture_data_from_file("sample_textures/venus.png");
    planets_textures_data[2] = make_image_texture_data_from_file("sample_textures/earth.png");
    planets_textures_data[3] = make_image_texture_data_from_file("sample_textures/mars.png");
    planets_textures_data[4] = make_image_texture_data_from_file("sample_textures/jupiter.png");

    for (int i = 0; i < N_SPHERES - 1; i++) {
        planets_textures[i] = make_image_texture(&planets_textures_data[i]);
        planets_material_data[i] = make_lambertian_material_data(&planets_textures[i]);
        materials[i] = make_lambertian_material(&planets_material_data[i]);
    }

    spheres[0] = make_sphere(make_vec3(0, 6.9, 5.7), 2.5, &materials[0]);
    spheres[1] = make_sphere(make_vec3(0, 6.9, 20.2), 6.0, &materials[1]);
    spheres[2] = make_sphere(make_vec3(0, 6.9, 36.5), 6.3, &materials[2]);
    spheres[3] = make_sphere(make_vec3(0, 6.9, 53), 3.4, &materials[3]);
    spheres[4] = make_sphere(make_vec3(0, 6.9, 70), 6.9, &materials[4]);

    MetalData* metal_data;
    checkCudaErrors(cudaMallocManaged(&metal_data, sizeof(*metal_data)));
    *metal_data = make_metal_material_data(make_color(0.2, 0.2, 0.2), 0.0);
    materials[N_SPHERES - 1] = make_metal_material(metal_data);
    spheres[N_SPHERES - 1] = make_sphere(make_vec3(0, -10000, 0), 10000, &materials[N_SPHERES - 1]);

    scene.spheres = spheres;
    scene.spheres_length = N_SPHERES;

    return scene;
}