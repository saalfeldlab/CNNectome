import os
from math import floor, ceil
import json
import h5py
import numpy as np


def make_inference_mask(output_shape, output_file, mask_file):
    shape_full = [30000, 30000, 1875][::-1]
    downscale_factor = [128, 128, 13][::-1]

    # output_shape = [56] * 3
    output_shape_ds = [outs / ds for outs, ds in zip(output_shape, downscale_factor)]

    pred_mask_shape = [
        sfull // outs + 1 for sfull, outs in zip(shape_full, output_shape)
    ]

    assert os.path.exists(mask_file), mask_file
    with h5py.File(mask_file) as f:
        ds = f["data"]
        mask = ds[:]

    prediction_mask = np.zeros(pred_mask_shape, dtype="uint8")
    prediction_blocks = []
    # generate blocks
    for z in range(0, shape_full[0], output_shape[0]):
        print("generating for", z)
        for y in range(0, shape_full[1], output_shape[1]):
            for x in range(0, shape_full[2], output_shape[2]):
                z_ds = z / downscale_factor[0]
                y_ds = y / downscale_factor[1]
                x_ds = x / downscale_factor[2]

                stop_z = z_ds + output_shape_ds[0]
                stop_y = y_ds + output_shape_ds[1]
                stop_x = x_ds + output_shape_ds[2]

                bb = np.s_[
                    int(floor(z_ds)) : int(ceil(stop_z)),
                    int(floor(y_ds)) : int(ceil(stop_y)),
                    int(floor(x_ds)) : int(ceil(stop_x)),
                ]

                mask_block = mask[bb]

                if np.sum(mask_block) > 0:
                    prediction_blocks.append([z, y, x])
                    prediction_mask[
                        z // output_shape[0], y // output_shape[1], x // output_shape[2]
                    ] = 1

    with open(output_file, "w") as f:
        json.dump(prediction_blocks, f)


if __name__ == "__main__":
    output_shape = (71, 650, 650)
    output_file = (
        "/nrs/saalfeld/heinrichl/synapses/sampleE_DTU2_offsets/block_list_in_mask.json"
    )
    mask_file = "/nrs/saalfeld/heinrichl/data/sampleE/neuropil_mask_s7.h5"
    make_inference_mask(output_shape, output_file, mask_file)
