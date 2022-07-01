#include <geometry/sphere.h>
#include <materials/dielectric.h>
#include <materials/lambertian.h>
#include <materials/metal.h>
#include <scenes/test_scene.h>
#include <util/check_cuda_errors.h>

__constant__ LambertianData dev_matdata_ground;
__constant__ Material dev_material_ground;
__constant__ LambertianData dev_matdata_center;
__constant__ Material dev_material_center;
__constant__ DielectricData dev_matdata_left;
__constant__ Material dev_material_left;
__constant__ MetalData dev_matdata_right;
__constant__ Material dev_material_right;

static int const N_SPHERES = 4;

__constant__ Sphere dev_spheres[N_SPHERES];

#define DECLARE_PTR_TO_SYMBOL(Type, name)                             \
    Type* name;                                                       \
    checkCudaErrors(cudaGetSymbolAddress((void**)&name, dev_##name)); \
    checkCudaErrors(cudaDeviceSynchronize());

static void initialize_materials(void)
{
    struct LambertianData d;

    d = make_lambertian_material_data(make_color(0.8, 0.8, 0));
    checkCudaErrors(cudaMemcpyToSymbol(dev_matdata_ground, &d, sizeof(d)));
    checkCudaErrors(cudaDeviceSynchronize());

    d = make_lambertian_material_data(make_color(0.1, 0.2, 0.5));
    checkCudaErrors(cudaMemcpyToSymbol(dev_matdata_center, &d, sizeof(d)));
    checkCudaErrors(cudaDeviceSynchronize());

    struct DielectricData dielectric_data;
    dielectric_data = make_dielectric_material_data(0.7);
    checkCudaErrors(cudaMemcpyToSymbol(dev_matdata_left, &dielectric_data, sizeof(dielectric_data)));
    checkCudaErrors(cudaDeviceSynchronize());

    struct MetalData metal_data;
    metal_data = make_metal_material_data(make_color(0.7, 0.6, 0.5), 0.1);
    checkCudaErrors(cudaMemcpyToSymbol(dev_matdata_right, &metal_data, sizeof(d)));
    checkCudaErrors(cudaDeviceSynchronize());

    DECLARE_PTR_TO_SYMBOL(LambertianData, matdata_ground);
    DECLARE_PTR_TO_SYMBOL(LambertianData, matdata_center);
    DECLARE_PTR_TO_SYMBOL(DielectricData, matdata_left);
    DECLARE_PTR_TO_SYMBOL(MetalData, matdata_right);

    Material material;
    material = make_lambertian_material(matdata_ground);
    checkCudaErrors(cudaMemcpyToSymbol(dev_material_ground, &material, sizeof(material)));
    checkCudaErrors(cudaDeviceSynchronize());

    material = make_lambertian_material(matdata_center);
    checkCudaErrors(cudaMemcpyToSymbol(dev_material_center, &material, sizeof(material)));
    checkCudaErrors(cudaDeviceSynchronize());

    material = make_dielectric_material(matdata_left);
    checkCudaErrors(cudaMemcpyToSymbol(dev_material_left, &material, sizeof(material)));
    checkCudaErrors(cudaDeviceSynchronize());

    material = make_metal_material(matdata_right);
    checkCudaErrors(cudaMemcpyToSymbol(dev_material_right, &material, sizeof(material)));
    checkCudaErrors(cudaDeviceSynchronize());
}

struct Scene create_test_scene(void)
{
    initialize_materials();

    DECLARE_PTR_TO_SYMBOL(Material, material_ground);
    DECLARE_PTR_TO_SYMBOL(Material, material_center);
    DECLARE_PTR_TO_SYMBOL(Material, material_left);
    DECLARE_PTR_TO_SYMBOL(Material, material_right);

    Sphere s[N_SPHERES];
    s[0] = make_sphere(make_vec3(0, -100.5, 1), 100, material_ground);
    s[1] = make_sphere(make_vec3(0, 0, 1), 0.5, material_center);
    s[2] = make_sphere(make_vec3(1, 0, 1), 0.5, material_left);
    s[3] = make_sphere(make_vec3(-1, 0, 1), 0.5, material_right);
    checkCudaErrors(cudaMemcpyToSymbol(dev_spheres, s, sizeof(s)));
    checkCudaErrors(cudaDeviceSynchronize());

    Scene scene;
    scene.spheres_length = N_SPHERES;
    checkCudaErrors(cudaGetSymbolAddress((void**)&scene.spheres, dev_spheres));
    checkCudaErrors(cudaDeviceSynchronize());

    scene.spheres_length = 4;

    return scene;
}