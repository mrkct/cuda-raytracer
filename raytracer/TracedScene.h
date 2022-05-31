#ifndef TRACED_SCENE_H
#define TRACED_SCENE_H

#include <cstdint>


class TracedScene
{
    public:
    int width() const { return 0; }
    int height() const { return 0; }
    uint8_t const* pixel_data() const { return nullptr; }
};

#endif