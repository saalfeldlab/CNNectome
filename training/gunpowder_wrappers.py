from __future__ import print_function
import h5py
import os
from gunpowder import *


def prepare_h5source(
    ds_dir,
    volume_name,
    label_file=("groundtruth_seg.h5", "main"),
    raw_file=("im_uint8.h5", "main"),
    add_dummy_mask=True,
):
    h5_filepath = ".{}.h5".format(volume_name)
    if add_dummy_mask:
        with h5py.File(
            os.path.join(ds_dir, volume_name, label_file[0]), "r"
        ) as f_labels:
            mask_shape = f_labels[label_file[1]].shape
    with h5py.File(h5_filepath, "w") as h5:
        h5["volumes/raw"] = h5py.ExternalLink(
            os.path.join(ds_dir, volume_name, raw_file[0]), raw_file[1]
        )
        h5["volumes/labels/neuron_ids"] = h5py.ExternalLink(
            os.path.join(ds_dir, volume_name, label_file[0]), label_file[1]
        )
        datasets = {
            VolumeTypes.RAW: "volumes/raw",
            VolumeTypes.GT_LABELS: "volumes/labels/neuron_ids",
        }
        if add_dummy_mask:
            h5.create_dataset(
                name="volumes/labels/mask", dtype="uint8", shape=mask_shape, fillvalue=1
            )
            datasets[VolumeTypes.GT_MASK] = "volumes/labels/mask"

    h5source = gunpowder.Hdf5Source(
        h5_filepath,
        datasets=datasets,
        # resolutions=(8, 8, 8),
    )
    return h5source
