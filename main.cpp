#include <chrono>
#include <iostream>
#include <raytracer/Camera.h>
#include <raytracer/Raytracer.h>
#include <raytracer/geometry/Sphere.h>
#include <raytracer/scenes/TestScene.h>
#include <string.h>
#include <string>

struct Args {
    char const* output_path;
    int image_width, image_height;

    static Args parse(int argc, char** argv)
    {
#define OPT_WITH_ARG(short, long) \
    ((strcmp(argv[i], short) == 0 || strcmp(argv[i], long) == 0) && i + 1 < argc)

        auto const& parse_positive_integer_or_exit = [](char const* arg, char const* argname) {
            try {
                int x = std::stoi(arg);
                if (x <= 0)
                    throw std::invalid_argument("");

                return x;
            } catch (std::invalid_argument&) {
                std::cerr
                    << "Invalid value for argument " << argname << ". "
                    << "I expected a positive integer, got '" << arg << "' instead." << std::endl;
                std::exit(-2);
            }
        };

        Args args = {
            .output_path = "output.png",
            .image_width = 640,
            .image_height = 480
        };

        for (int i = 1; i < argc; i++) {
            if (OPT_WITH_ARG("-o", "--output")) {
                args.output_path = argv[++i];
            } else if (OPT_WITH_ARG("-w", "--width")) {
                args.image_width = parse_positive_integer_or_exit(argv[++i], "--width");
            } else if (OPT_WITH_ARG("-h", "--height")) {
                args.image_height = parse_positive_integer_or_exit(argv[++i], "--height");
            }
        }

        return args;
    }
};

int main(int argc, char** argv)
{
    auto args = Args::parse(argc, argv);

    std::cout << "Starting render with" << std::endl
              << "\tSize: " << args.image_width << "x" << args.image_height << std::endl
              << "\tOutput: " << args.output_path << std::endl;

    auto raytracer = Raytracer(args.image_width, args.image_height);
#define DEG2RAD(d) (d * M_PI / 1.80f)
    auto render_start_time = std::chrono::high_resolution_clock::now();
    auto frame_only_time = std::chrono::milliseconds();
    for (int i = 0; i < 180; i += 30) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        auto traced_scene = raytracer.trace_scene(
            { 2.0 * cosf(DEG2RAD(i)), 0, -1.0 }, { 0, 0, -1 },
            TestScene::init);
        auto frame_finish_time = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_finish_time - frame_start_time);

        std::cout << "Frame " << i << "\tTime: " << frame_time.count() << "ms" << std::endl;
        frame_only_time += frame_time;

        auto frame_id = i < 10 ? "0" + std::to_string(i) : std::to_string(i);
        auto output_path = std::string(args.output_path) + frame_id + ".png";
        std::cout << "Writing frame to " << output_path << std::endl;
        auto rc = traced_scene.write_to_file(output_path.c_str());
        if (rc) {
            std::cerr << "Failed to write frame " << i << " to disk due to error code " << rc << std::endl;
        }
    }

    auto total_rendering_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - render_start_time);
    std::cout << "Render finished" << std::endl
              << "\tTotal Time:\t" << total_rendering_time.count() << std::endl
              << "\tRender-only Time:\t" << frame_only_time.count() << std::endl;

    return 0;
}