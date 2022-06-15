#ifndef TRACED_SCENE_H
#define TRACED_SCENE_H

#include <cstdint>
#include <memory>

class TracedScene {
public:
    TracedScene(unsigned int width, unsigned int height, uint32_t* data)
        : m_width(width)
        , m_height(height)
        , m_data(data)
    {
    }

    int width() const { return m_width; }
    int height() const { return m_height; }
    static int bytes_per_pixel() { return 4; }
    uint32_t const* pixel_data() const { return m_data.get(); }

private:
    unsigned int m_width, m_height;
    std::unique_ptr<uint32_t[]> m_data;
};

#endif