#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <stdint.h>
#include <util/vec3.h>

struct Framebuffer {
    color* color_data;
    uint32_t* data;
    unsigned byte_size;
    unsigned width, height;
};

/**
 * @brief Allocates a color framebuffer and an ARGB framebuffer; the color
 * framebuffer is allocated in device memory, and therefore only accessible
 * by the GPU, while the ARGB framebuffer is allocated in managed memory
 *
 * @param width Width of the framebuffer
 * @param height Height of the framebuffer
 * @return struct Framebuffer
 */
struct Framebuffer alloc_framebuffer(unsigned width, unsigned height);
void write_framebuffer_to_file(struct Framebuffer, char const* filename);

#endif