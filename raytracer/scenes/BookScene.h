#pragma once

#include <raytracer/HittableList.h>
#include <raytracer/geometry/Hittable.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/material/Dielectric.h>
#include <raytracer/material/Lambertian.h>
#include <raytracer/material/Metal.h>
#include <raytracer/util/CudaHelpers.h>
#include <raytracer/util/DeviceRNG.h>
/*
class BookScene : public Hittable {
public:
    static Hittable& init(DeviceRNG& rng) { return new_on_device<BookScene>(rng); }

    __device__ BookScene(DeviceRNG& rng)
        : m_world(*new HittableList)
    {
        m_world.reserve(30);

        auto& glass = *new Dielectric(rng, 1.5f);
        auto& lambertian = *new Lambertian(rng, Color(0.4f, 0.2f, 0.1f));
        auto& metal = *new Metal(rng, Color(0.7f, 0.6f, 0.5f), 0.0f);
        auto& ground_material = *new Lambertian(rng, Color(0.5f, 0.5f, 0.5f));

        static constexpr int colors = 10;
        auto** colored_lambertians = new Lambertian*[colors];
        auto** colored_metals = new Metal*[colors];
        for (int i = 0; i < colors; i++) {
            colored_lambertians[i] = new Lambertian(rng, Color(rng.next(0), rng.next(0), rng.next(0)));
            colored_metals[i] = new Metal(rng, Color(rng.next(0), rng.next(0), rng.next(0)), rng.next(0, 0, 0.5));
        }

        m_world.append(new Sphere({ 0, -1000, 0 }, 1000, ground_material));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = rng.next(0);
                Point3 center(a + 0.9 * rng.next(0), 0.2, b + 0.9 * rng.next(0));

                if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                    Material* sphere_material;

                    if (choose_mat < 0.8) {
                        sphere_material = colored_lambertians[(int)rng.next(0, 0, colors)];
                        m_world.append(new Sphere(center, 0.2, *sphere_material));
                    } else if (choose_mat < 0.95) {
                        sphere_material = colored_metals[(int)rng.next(0, 0, colors)];
                        m_world.append(new Sphere(center, 0.2, *sphere_material));
                    } else {
                        m_world.append(new Sphere(center, 0.2, glass));
                    }
                }
            }
        }

        m_world.append(new Sphere({ 0, 1, 0 }, 1.0, glass));
        m_world.append(new Sphere({ -4, 1, 0 }, 1.0, lambertian));
        m_world.append(new Sphere({ 4, 1, 0 }, 1.0, metal));
    }

    __device__ virtual bool hit(Ray const& r, float t_min, float t_max, HitRecord& rec) const override
    {
        return m_world.hit(r, t_min, t_max, rec);
    }

private:
    HittableList& m_world;
};
*/