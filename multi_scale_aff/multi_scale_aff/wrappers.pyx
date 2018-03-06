import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

#ctypedef np.int64_t LABELTYPE_t

cdef extern from "multi_scale_aff.hxx":
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

def compute_multi_scale_affinities(np.ndarray[int64_t, ndim=3] labels_arr, tuple blocking):

    if not labels_arr.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous points arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        labels_arr = np.ascontiguousarray(labels_arr)

    cdef int blocking_ax0 = blocking[0]
    cdef int blocking_ax1 = blocking[1]
    cdef int blocking_ax2 = blocking[2]

    cdef int labels_shape_ax0 = labels_arr.shape[0]
    cdef int labels_shape_ax1 = labels_arr.shape[1]
    cdef int labels_shape_ax2 = labels_arr.shape[2]
    new_shape = tuple(3,
                      labels_arr.shape[0]/blocking[0] -1,
                      labels_arr.shape[1]/blocking[1]-1,
                      labels_arr.shape[2]/blocking[2]-1)
    cdef np.ndarray[double, ndim=4] affinities = np.zeros(new_shape, dtype=np.float64)

    compute_multi_scale_affinities_impl(&labels_arr[0,0,0], &affinities[0,0,0,0],
                                        blocking_ax0, blocking_ax1, blocking_ax2,
                                        labels_shape_ax0, labels_shape_ax1, labels_shape_ax2)
    return affinities

