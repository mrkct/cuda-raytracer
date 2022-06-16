#include <raytracer/HittableList.h>

__device__ void HittableList::reserve(size_t count)
{
    m_objects.elements = new Hittable*[count];
}

__device__ bool HittableList::hit(Ray const& r, double t_min, double t_max, HitRecord& rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (size_t i = 0; i < m_objects.length; i++) {
        auto const* object = m_objects[i];
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
