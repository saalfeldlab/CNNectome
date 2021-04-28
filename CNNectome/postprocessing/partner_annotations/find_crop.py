import zarr
import os
import numpy as np
from .find_partners import bbox_ND
import h5py


def find_crop(filename_src, dataset_src, bg_label=0xFFFFFFFFFFFFFFFD):
    if filename_src.endswith(".hdf") or filename_src.endswith(".h5"):
        srcf = h5py.File(filename_src, "r")
    else:
        srcf = zarr.open(filename_src, mode="r")
    bb = bbox_ND(srcf[dataset_src][:] != bg_label)
    print(srcf[dataset_src].shape)
    off = (bb[0], bb[2], bb[4])
    shape = (bb[1] - bb[0] + 1, bb[3] - bb[2] + 1, bb[5] - bb[4] + 1)
    return off, shape
