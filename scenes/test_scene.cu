#include <geometry/sphere.h>
#include <materials/dielectric.h>
#include <materials/lambertian.h>
#include <materials/metal.h>
#include <scenes/test_scene.h>
#include <textures/solid_texture.h>
#include <textures/texture.h>
#include <util/check_cuda_errors.h>

__constant__ Texture dev_ground_texture;
__constant__ Texture dev_center_texture;

__constant__ SolidTextureData dev_ground_texture_data;
__constant__ SolidTextureData dev_center_texture_data;

__constant__ Material dev_ground_material;
__constant__ Material dev_left_material;
__constant__ Material dev_center_material;
__constant__ Material dev_right_material;

__constant__ LambertianData dev_ground_material_data;
__constant__ DielectricData dev_left_material_data;
__constant__ LambertianData dev_center_material_data;
__constant__ MetalData dev_right_material_data;

static int const N_SPHERES = 4;

__constant__ Sphere dev_spheres[N_SPHERES];

#define DECLARE_PTR_TO_SYMBOL(Type, name)                             \
    Type* name;                                                       \
    checkCudaErrors(cudaGetSymbolAddress((void**)&name, dev_##name)); \
    checkCudaErrors(cudaDeviceSynchronize());

static void initialize_textures(void)
{
    struct Texture texture;
    struct SolidTextureData texture_data;

    {
        // Ground
        texture_data = make_solid_texture_data(make_color(0.8, 0.8, 0));
        checkCudaErrors(cudaMemcpyToSymbol(dev_center_texture_data, &texture_data, sizeof(texture_data)));
        checkCudaErrors(cudaDeviceSynchronize());

        DECLARE_PTR_TO_SYMBOL(SolidTextureData, ground_texture_data);

        texture = make_solid_texture(ground_texture_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_ground_texture, &texture, sizeof(texture)));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // Center
        texture_data = make_solid_texture_data(make_color(0.1, 0.2, 0.5));
        checkCudaErrors(cudaMemcpyToSymbol(dev_center_texture_data, &texture_data, sizeof(texture_data)));
        checkCudaErrors(cudaDeviceSynchronize());

        DECLARE_PTR_TO_SYMBOL(SolidTextureData, center_texture_data);

        texture = make_solid_texture(center_texture_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_center_texture, &texture, sizeof(texture)));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

static void initialize_materials(void)
{
    {
        // Ground
        struct LambertianData data;
        DECLARE_PTR_TO_SYMBOL(Texture, ground_texture);

        data = make_lambertian_material_data(ground_texture);
        checkCudaErrors(cudaMemcpyToSymbol(dev_ground_material_data, &data, sizeof(data)));
        checkCudaErrors(cudaDeviceSynchronize());

        struct Material material;
        DECLARE_PTR_TO_SYMBOL(LambertianData, ground_material_data);

        material = make_lambertian_material(ground_material_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_ground_material, &material, sizeof(material)));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // Center
        struct LambertianData data;
        DECLARE_PTR_TO_SYMBOL(Texture, center_texture);

        data = make_lambertian_material_data(center_texture);
        checkCudaErrors(cudaMemcpyToSymbol(dev_center_material_data, &data, sizeof(data)));
        checkCudaErrors(cudaDeviceSynchronize());

        struct Material material;
        DECLARE_PTR_TO_SYMBOL(LambertianData, center_material_data);

        material = make_lambertian_material(center_material_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_center_material, &material, sizeof(material)));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // Left
        struct DielectricData dielectric_data;
        dielectric_data = make_dielectric_material_data(0.7);
        checkCudaErrors(cudaMemcpyToSymbol(dev_left_material_data, &dielectric_data, sizeof(dielectric_data)));
        checkCudaErrors(cudaDeviceSynchronize());

        DECLARE_PTR_TO_SYMBOL(DielectricData, left_material_data);

        struct Material material;
        material = make_dielectric_material(left_material_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_left_material, &material, sizeof(material)));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        // Right
        struct MetalData metal_data;
        metal_data = make_metal_material_data(make_color(0.7, 0.6, 0.5), 0.1);
        checkCudaErrors(cudaMemcpyToSymbol(dev_right_material_data, &metal_data, sizeof(metal_data)));
        checkCudaErrors(cudaDeviceSynchronize());

        DECLARE_PTR_TO_SYMBOL(MetalData, right_material_data);

        struct Material material;
        material = make_metal_material(right_material_data);
        checkCudaErrors(cudaMemcpyToSymbol(dev_right_material, &material, sizeof(material)));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

static void initialize_spheres(void)
{
    DECLARE_PTR_TO_SYMBOL(Material, ground_material);
    DECLARE_PTR_TO_SYMBOL(Material, center_material);
    DECLARE_PTR_TO_SYMBOL(Material, left_material);
    DECLARE_PTR_TO_SYMBOL(Material, right_material);

    struct Sphere s[N_SPHERES];
    s[0] = make_sphere(make_vec3(0, -100.5, 1), 100, ground_material);
    s[1] = make_sphere(make_vec3(0, 0, 1), 0.5, center_material);
    s[2] = make_sphere(make_vec3(1, 0, 1), 0.5, left_material);
    s[3] = make_sphere(make_vec3(-1, 0, 1), 0.5, right_material);
    checkCudaErrors(cudaMemcpyToSymbol(dev_spheres, s, sizeof(s)));
    checkCudaErrors(cudaDeviceSynchronize());
}

struct Scene create_test_scene(void)
{
    initialize_textures();
    initialize_materials();
    initialize_spheres();

    Scene scene;
    scene.spheres_length = N_SPHERES;
    checkCudaErrors(cudaGetSymbolAddress((void**)&scene.spheres, dev_spheres));
    checkCudaErrors(cudaDeviceSynchronize());

    scene.spheres_length = 4;

    return scene;
}