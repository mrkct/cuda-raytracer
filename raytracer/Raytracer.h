#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "TracedScene.h"
#include "Scene.h"

class Raytracer {
public:
    Raytracer(int image_width, int image_height) {}

    TracedScene trace_scene(Scene const&) {
        return {};
    }
};

#endif