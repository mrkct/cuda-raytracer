#ifndef ARG_PARSING_H
#define ARG_PARSING_H

struct Args {
    char const* output_path;
    unsigned image_width, image_height;
    int frames;
    int samples;
};

/**
 * @brief Parses the command line arguments and returns a struct
 * initialized with either default values or the user provided ones
 *
 * @param argc The main's argc
 * @param argv The main's argv
 * @return struct Args
 */
struct Args parse_args(int argc, char const* argv[]);

/**
 * @brief Converts 'arg' into a positive integer, if that fails
 * prints an error message using 'argument_name' and exists the program
 *
 * @param arg The string to be parsed into an integer
 * @param argument_name Name of the argument to be printed in the error message
 * @return unsigned
 */
unsigned parse_positive_integer_or_exit(char const* arg, char const* argument_name);

#endif