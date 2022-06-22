#include <raytracer/geometry/Sphere.h>

__device__ bool Sphere::hit(Ray const& ray, float t_min, float t_max, HitRecord& rec) const
{
    Vec3 oc = ray.origin() - m_center;
    auto a = ray.direction().length_squared();
    auto half_b = dot(oc, ray.direction());
    auto c = oc.length_squared() - m_radius * m_radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = ray.at(rec.t);
    rec.material = &m_material;
    auto outward_normal = (rec.p - m_center) / m_radius;
    rec.set_face_normal(ray, outward_normal);

    return true;
}