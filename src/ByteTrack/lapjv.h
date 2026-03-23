#ifndef LAPJV_H
#define LAPJV_H
#pragma once

#include <cstddef>

namespace byte_track
{
    int lapjv_internal(const size_t n, double *cost[], int *x, int *y);
}

#endif // LAPJV_H
