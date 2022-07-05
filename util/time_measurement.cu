#include <sys/time.h>
#include <util/time_measurement.h>

long long current_time_in_microseconds(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    static suseconds_t biggest = 0;
    biggest = biggest < t.tv_usec ? t.tv_usec : biggest;

    return biggest;
}