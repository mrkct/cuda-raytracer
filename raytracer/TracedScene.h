#ifndef TRACED_SCENE_H
#define TRACED_SCENE_H

#include <cstdint>
#include <memory>


class TracedScene
{
    public:
    TracedScene(int width, int height, uint8_t *data)
        : m_width(width), m_height(height), m_data(data)
    {}

    int width() const { return m_width; }
    int height() const { return m_height; }
    static int bytes_per_pixel() { return 3; }
    uint8_t const* pixel_data() const { return m_data.get(); }

    private:
    int m_width, m_height;
    std::unique_ptr<uint8_t[]> m_data;
};

#endif