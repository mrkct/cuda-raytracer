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

struct Framebuffer alloc_framebuffer(unsigned width, unsigned height);
void write_framebuffer_to_file(struct Framebuffer, char const* filename);

#endif