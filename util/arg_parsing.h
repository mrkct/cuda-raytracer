#ifndef ARG_PARSING_H
#define ARG_PARSING_H

struct Args {
    char const* output_path;
    unsigned image_width, image_height;
    int frames;
};

struct Args parse_args(int argc, char const* argv[]);
unsigned parse_positive_integer_or_exit(char const* arg, char const* argument_name);

#endif