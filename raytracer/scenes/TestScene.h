#pragma once

#include <raytracer/HittableList.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Dielectric.h>
#include <raytracer/material/Lambertian.h>
#include <raytracer/material/Metal.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>

__constant__ uint8_t g_ground_material_storage[sizeof(Lambertian)];
__constant__ uint8_t g_center_material_storage[sizeof(Lambertian)];
__constant__ uint8_t g_left_material_storage[sizeof(Dielectric)];
__constant__ uint8_t g_right_material_storage[sizeof(Metal)];

__global__ void create_lambertian(Lambertian* l, Color albedo)
{
    new (l) Lambertian(albedo);
}
__global__ void create_metal(Metal* m, Color albedo, float fuzz)
{
    new (m) Metal(albedo, fuzz);
}
__global__ void create_dielectric(Dielectric* d, float r)
{
    new (d) Dielectric(r);
}

class TestScene : public Hittable {
public:
    static Hittable& init(DeviceRNG::Builder)
    {
        {
            Lambertian* l;
            checkCudaErrors(cudaMalloc(&l, sizeof(Lambertian)));
            create_lambertian<<<1, 1>>>(l, Color(0.8, 0.8, 0));
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpyToSymbol(g_ground_material_storage, l, sizeof(*l), 0, cudaMemcpyDeviceToDevice));

            create_lambertian<<<1, 1>>>(l, Color(0.1, 0.2, 0.5));
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpyToSymbol(g_center_material_storage, l, sizeof(*l), 0, cudaMemcpyDeviceToDevice));

            checkCudaErrors(cudaFree(l));
        }
        {
            Dielectric* d;
            checkCudaErrors(cudaMalloc(&d, sizeof(Dielectric)));
            create_dielectric<<<1, 1>>>(d, 0.5);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpyToSymbol(g_left_material_storage, d, sizeof(*d), 0, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaFree(d));
        }
        {
            Metal* m;
            checkCudaErrors(cudaMalloc(&m, sizeof(Metal)));
            create_metal<<<1, 1>>>(m, Color(0.8, 0.6, 0.2), 0.0);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpyToSymbol(g_right_material_storage, m, sizeof(*m), 0, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaFree(m));
        }
        return new_on_device<TestScene>();
    }

    __device__ TestScene()
        : m_world(*new HittableList)
    {
        auto& material_ground = *reinterpret_cast<Lambertian*>(g_ground_material_storage);
        auto& material_center = *reinterpret_cast<Lambertian*>(g_center_material_storage);
        auto& material_left = *reinterpret_cast<Dielectric*>(g_left_material_storage);
        auto& material_right = *reinterpret_cast<Metal*>(g_right_material_storage);

        printf("name: %d\n", material_ground.x);

        m_world.reserve(4);

        m_world.append(new Sphere({ 0, 0, -1 }, 0.5, material_center));
        m_world.append(new Sphere({ 1, 0, -1 }, 0.5, material_right));
        m_world.append(new Sphere({ -1, 0, -1 }, 0.5, material_left));

        m_world.append(new Sphere({ 0, -100.5, -1 }, 100, material_ground));
    }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

private:
    HittableList& m_world;
};