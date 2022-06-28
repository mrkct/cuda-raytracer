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

__constant__ uint8_t g_spheres_storage[sizeof(Sphere) * 4];
__constant__ Hittable* g_world[4];

static __global__ void create_lambertian(Lambertian* l, Color albedo)
{
    new (l) Lambertian(albedo);
}
static __global__ void create_metal(Metal* m, Color albedo, float fuzz)
{
    new (m) Metal(albedo, fuzz);
}
static __global__ void create_dielectric(Dielectric* d, float r)
{
    new (d) Dielectric(r);
}
static __global__ void create_sphere(Sphere* s, Point3 center, float radius, Material const& material)
{
    printf("sphere @ %p\n", s);
    new (s) Sphere(center, radius, material);
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

        auto& material_ground = *reinterpret_cast<Lambertian*>(g_ground_material_storage);
        auto& material_center = *reinterpret_cast<Lambertian*>(g_center_material_storage);
        auto& material_left = *reinterpret_cast<Dielectric*>(g_left_material_storage);
        auto& material_right = *reinterpret_cast<Metal*>(g_right_material_storage);

        {
            Sphere* p;
            checkCudaErrors(cudaGetSymbolAddress((void**)&p, g_spheres_storage));
            Hittable* w[] = { &p[0], &p[1], &p[2], &p[3] };
            cudaMemcpyToSymbol(g_world, w, sizeof(w));
            printf("a: %p     b: %p      c: %p       d: %p\n", w[0], w[1], w[2], w[3]);

            int const SPHERES_N = 4;
            Sphere* s;
            checkCudaErrors(cudaMalloc(&s, sizeof(Sphere) * SPHERES_N));

            create_sphere<<<1, 1>>>(&s[0], Vec3(0, 0, -1), 0.5, material_center);
            create_sphere<<<1, 1>>>(&s[1], Vec3(1, 0, -1), 0.5, material_right);
            create_sphere<<<1, 1>>>(&s[2], Vec3(-1, 0, -1), 0.5, material_left);
            create_sphere<<<1, 1>>>(&s[3], Vec3(0, -100.5, -1), 100, material_ground);
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemcpyToSymbol(
                g_spheres_storage,
                s,
                sizeof(Sphere) * SPHERES_N,
                0,
                cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaFree(s));
        }

        return new_on_device<TestScene>();
    }

    __device__
    TestScene()
        : m_world(*new HittableList(DeviceArray<Hittable*> { .elements = g_world, .length = 4 }))
    {
        printf("x0 sphere @ %p\n", m_world.m_objects[0]);
        printf("x1 sphere @ %p\n", m_world.m_objects[1]);
        printf("x2 sphere @ %p\n", m_world.m_objects[2]);

        printf("id: %d\n", m_world.m_objects[0]->id());
    }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

    __device__ virtual int id() const override { return -2; };

private:
    HittableList& m_world;
};