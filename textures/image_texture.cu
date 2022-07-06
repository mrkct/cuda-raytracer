#include <stdio.h>
#include <stdlib.h>
#include <textures/image_texture.h>
#include <util/check_cuda_errors.h>
#include <util/stb_image.h>

struct ImageTextureData make_image_texture_data_from_file(char const* filename)
{
    int w, h;
    unsigned char* host_image_data = stbi_load(filename, &w, &h, NULL, 4);
    if (!host_image_data) {
        printf("failed to load %s : %s\n", filename, stbi_failure_reason());
        exit(-1);
    }
    uint32_t* dev_image_data;
    int const image_size_in_bytes = 4 * w * h;
    checkCudaErrors(cudaMalloc(&dev_image_data, image_size_in_bytes));
    checkCudaErrors(cudaMemcpy(dev_image_data, host_image_data, image_size_in_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    stbi_image_free(host_image_data);

    return (struct ImageTextureData) { .data = dev_image_data, .width = (unsigned)w, .height = (unsigned)h };
}

struct ImageTextureData make_image_texture_data(uint32_t* data, unsigned width, unsigned height)
{
    return (struct ImageTextureData) { .data = data, .width = width, .height = height };
}

struct Texture make_image_texture(struct ImageTextureData* d)
{
    return (struct Texture) { .texture_type = IMAGE, .data = d };
}

__device__ color image_texture_color_at(struct ImageTextureData* d, float const u, float const v, point3 const&)
{
    int const x = u * (d->width - 1);
    int const y = (d->height - 1) - v * (d->height - 1);
    return rgba_to_color(d->data[y * d->width + x]);
}
