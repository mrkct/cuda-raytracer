#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H

#include <stdio.h>

#define FMT_3F "%f\t%f\t%f"
#define FMT_XYZ(v) v.x, v.y, v.z
#define tprintf  \
    if (id == 0) \
    printf

#endif