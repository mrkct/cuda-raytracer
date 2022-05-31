#include <raytracer/Raytracer.h>
#include <stdio.h>


__global__ void helloFromGPU (void) {
    printf("hello world from the GPU\n");
}

TracedScene Raytracer::trace_scene(Scene const&)
{
    helloFromGPU<<<1, 10>>>();
    return {};
}