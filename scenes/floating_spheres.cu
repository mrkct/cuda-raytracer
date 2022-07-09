#include <materials/dielectric.h>
#include <materials/lambertian.h>
#include <materials/metal.h>
#include <scenes/floating_spheres.h>
#include <textures/checkered_texture.h>
#include <textures/image_texture.h>
#include <textures/solid_texture.h>
#include <util/check_cuda_errors.h>

static int const N_SPHERES = 7;

struct Scene create_floating_spheres_scene(void)
{
    struct Scene scene;

    Sphere* spheres;
    checkCudaErrors(cudaMallocManaged(&spheres, sizeof(Sphere) * N_SPHERES));

    Material* materials;
    checkCudaErrors(cudaMallocManaged(&materials, sizeof(Material) * N_SPHERES));

    {
        LambertianData* lambertian_datas;
        checkCudaErrors(cudaMallocManaged(&lambertian_datas, sizeof(*lambertian_datas) * 4));

        Texture* textures;
        checkCudaErrors(cudaMallocManaged(&textures, sizeof(Texture) * 4));

        ImageTextureData* image_textures;
        checkCudaErrors(cudaMallocManaged(&image_textures, sizeof(ImageTextureData) * 1));
        image_textures[0] = make_image_texture_data_from_file("sample_textures/mars.png");
        textures[0] = make_image_texture(&image_textures[0]);

        CheckeredTextureData* checkered_texture;
        checkCudaErrors(cudaMallocManaged(&checkered_texture, sizeof(*checkered_texture) * 1));
        checkered_texture[0] = make_checkered_texture_data(0.1, make_color(1, 0, 0), make_color(1, 1, 1));
        textures[1] = make_checkered_texture(&checkered_texture[0]);

        SolidTextureData* solid_texture;
        checkCudaErrors(cudaMallocManaged(&solid_texture, sizeof(*solid_texture) * 2));
        solid_texture[0] = make_solid_texture_data(make_color(0.7, 0.2, 0.2));
        solid_texture[1] = make_solid_texture_data(make_color(0.2, 0.2, 0.7));
        textures[2] = make_solid_texture(&solid_texture[0]);
        textures[3] = make_solid_texture(&solid_texture[1]);

        for (int i = 0; i < 4; i++) {
            lambertian_datas[i] = make_lambertian_material_data(&textures[i]);
            materials[i] = make_lambertian_material(&lambertian_datas[i]);
        }
    }

    {
        MetalData* metal_data;
        checkCudaErrors(cudaMallocManaged(&metal_data, sizeof(*metal_data) * 2));

        metal_data[0] = make_metal_material_data(make_color(0.3, 0.3, 0.3), 0.05);
        metal_data[1] = make_metal_material_data(make_color(0.6, 0.5, 0.3), 0.4);

        materials[4] = make_metal_material(&metal_data[0]);
        materials[5] = make_metal_material(&metal_data[1]);
    }

    {
        DielectricData* dielectric_data;
        checkCudaErrors(cudaMallocManaged(&dielectric_data, sizeof(*dielectric_data) * 1));

        dielectric_data[0] = make_dielectric_material_data(0.9);

        materials[6] = make_dielectric_material(&dielectric_data[0]);
    }

    spheres[0] = make_sphere(make_vec3(-5, 4.5, -3), 2.5, &materials[0]);
    spheres[1] = make_sphere(make_vec3(6, 2, -6), 4.0, &materials[1]);
    spheres[2] = make_sphere(make_vec3(-5, 6.9, 2), 1.5, &materials[2]);
    spheres[3] = make_sphere(make_vec3(-2, 6.8, 6), 6.5, &materials[3]);
    spheres[4] = make_sphere(make_vec3(-1, 9, 9), 1.8, &materials[4]);
    spheres[5] = make_sphere(make_vec3(0, -3, 6), 2.7, &materials[5]);
    spheres[6] = make_sphere(make_vec3(-3, -4, -6), 4.5, &materials[6]);

    scene.spheres = spheres;
    scene.spheres_length = N_SPHERES;

    return scene;
}