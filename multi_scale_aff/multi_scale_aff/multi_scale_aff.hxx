#pragma once
#include <cstdint>

void compute_multi_scale_affinities_impl(
    const int64_t* labels,
    double* affinities,
    const int blocking_ax0,
    const int blocking_ax1,
    const int blocking_ax2,
    const int labels_shape_ax0,
    const int labels_shape_ax1,
    const int labels_shape_ax2
);
