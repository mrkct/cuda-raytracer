#include "stb_image_write.h"
#include "raytracer/Scene.h"
#include "raytracer/Raytracer.h"
#include <iostream>
#include <chrono>


struct Args {
    char const *output_path;
    int image_width, image_height;

    static Args parse(int argc, char **argv) {
        return Args {.output_path = "ciao", .image_width = 640, .image_height = 480};
    }
};

int main(int argc, char **argv)
{
    auto args = Args::parse(argc, argv);

    std::cout << "Starting render with" << std::endl
        << "\tSize: " << args.image_width << "x" << args.image_height << std::endl
        << "\tOutput: " << args.output_path << std::endl;

    Scene scene;
    // TODO: Prepare the scene

    auto raytracer = Raytracer(args.image_width, args.image_height);
    
    auto start_time = std::chrono::high_resolution_clock::now();;    
    auto traced_scene = raytracer.trace_scene(scene);
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time);

    std::cout << "Elapsed time: " << elapsed_time.count() << "ns" << std::endl;

    return stbi_write_png(
        args.output_path, 
        traced_scene.width(), 
        traced_scene.height(),
        4,
        traced_scene.pixel_data(),
        traced_scene.width() * 4
    );
}