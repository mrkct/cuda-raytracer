#include <materials/dielectric.h>
#include <materials/lambertian.h>
#include <materials/material.h>
#include <materials/metal.h>
#include <stdio.h>

__device__ bool material_scatter(struct Material const* material, curandState_t* rng_state, struct Ray ray,
    HitRecord* rec, color* out_attenuation, struct Ray* out_scattered)
{
    switch (material->material_type) {
    case LAMBERTIAN:
        return lambertian_scatter(material->data, rng_state, ray, rec, out_attenuation, out_scattered);
    case METAL:
        return metal_scatter(material->data, rng_state, ray, rec, out_attenuation, out_scattered);
    case DIELECTRIC:
        return dielectric_scatter(material->data, rng_state, ray, rec, out_attenuation, out_scattered);
    default:
        printf("NOT IMPLEMENTED!");
        return false;
    };
}
