#include <chrono>
#include <math.h>
#include <raytracer.h>
#include <scenes/planets.h>
#include <scenes/single_sphere.h>
#include <scenes/test_scene.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/arg_parsing.h>
#include <util/time_measurement.h>

#define MAX_PATH_LENGTH 512

void get_frame_output_path(char const* directory, int frame_index, char* buffer)
{
    sprintf(buffer, "%s_%09d.png", directory, frame_index);
}

int main(int argc, char const* argv[])
{
    struct Args args = parse_args(argc, argv);

    printf("Starting render with\n"
           "\tSize: %d x %d\n"
           "\tFrames: %d\n"
           "\tOutput folder: %s\n",
        args.image_width, args.image_height, args.frames, args.output_path);

    struct Framebuffer fb = alloc_framebuffer(args.image_width, args.image_height);
    struct Scene scene = create_planets_scene();

#define DEG2RAD(d) (d * M_PI / 180.f)

    char* output_path = (char*)malloc(sizeof(char) * MAX_PATH_LENGTH);
    long combined_render_only_time = 0;

    auto total_time_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.frames; i++) {
        auto render_time_start = std::chrono::high_resolution_clock::now();
        raytrace_scene(fb, scene, args.samples, PLANETS_CAMERA_FROM, PLANETS_CAMERA_LOOKAT, PLANETS_CAMERA_FOV);
        auto render_time_end = std::chrono::high_resolution_clock::now();
        auto render_time = std::chrono::duration_cast<std::chrono::microseconds>(render_time_end - render_time_start).count();

        printf("Frame %d took\t%ld us\n", i, render_time);
        combined_render_only_time += render_time;

        get_frame_output_path(args.output_path, i, output_path);
        printf("Writing frame %d to '%s'\n", i, output_path);
        write_framebuffer_to_file(fb, output_path);
    }

    auto total_time_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_time_end - total_time_start).count();

    printf("Render finished\n"
           "\tTotal Time: %ld us\n"
           "\tRender-only Time: %ld us\n",
        total_time, combined_render_only_time);

    return 0;
}