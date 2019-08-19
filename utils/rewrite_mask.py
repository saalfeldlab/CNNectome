import h5py
import numpy as np


def rewrite_mask(sample):
    print("Processing sample", sample)
    f = h5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{"
        "0:}_20160501.aligned.uncompressed.hdf".format(sample),
        "r",
    )
    g = h5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{"
        "0:}_cleftsorig_withvalidation.hdf".format(sample),
        "w",
    )

    clefts_orig = np.array(f["volumes/labels/clefts"])
    clefts_outside = np.sum(clefts_orig == 0xFFFFFFFFFFFFFFFF - 2)
    print("Clefts OUTSIDE", clefts_outside)
    mask = np.logical_not(
        np.logical_or(
            clefts_orig == 0xFFFFFFFFFFFFFFFF - 2, clefts_orig == 0xFFFFFFFFFFFFFFFF - 1
        )
    )
    print("Mask has dtype", mask.dtype)
    g.create_dataset(
        "volumes/masks/groundtruth", data=mask, dtype="float32", chunks=(26, 256, 256)
    )
    for k, v in f["volumes/masks/training"].attrs.items():
        g["volumes/masks/groundtruth"].attrs.create(k, v)

    neuron_ids = np.array(f["volumes/labels/neuron_ids"])
    neuron_ids[np.logical_not(mask)] = 0
    g.create_dataset(
        "volumes/labels/neuron_ids", data=neuron_ids, chunks=(26, 256, 256)
    )
    for k, v in f["volumes/labels/neuron_ids"].attrs.items():
        g["volumes/labels/neuron_ids"].attrs.create(k, v)
    del neuron_ids

    clefts = clefts_orig
    bg = clefts_orig == 0xFFFFFFFFFFFFFFFF
    clefts[bg] = 0
    clefts[np.logical_not(mask)] = 0
    clefts = (clefts > 0).astype(clefts_orig.dtype)
    print("Clefts has dtype", clefts.dtype)
    g.create_dataset("volumes/labels/clefts", data=clefts, chunks=(26, 256, 256))
    for k, v in f["volumes/labels/clefts"].attrs.items():
        g["volumes/labels/clefts"].attrs.create(k, v)
    del clefts_orig
    del clefts

    training_mask = np.zeros_like(mask)
    training_set = np.array(f["volumes/masks/training"])
    training_mask[training_set == 1] = 1
    training_mask[training_set == 0xFFFFFFFFFFFFFFFF] = 1
    # sanity_check = np.ones_like(mask)
    # sanity_check[]==0
    # clefts_orig[training_mask]=0
    assert clefts_outside == np.sum(training_set == 0xFFFFFFFFFFFFFFFF - 2)
    g.create_dataset(
        "volumes/masks/training",
        data=training_mask,
        dtype="float32",
        chunks=(26, 256, 256),
    )
    for k, v in f["volumes/masks/training"].attrs.items():
        g["volumes/masks/training"].attrs.create(k, v)
    del training_mask
    del training_set
    validation_mask = np.zeros_like(mask)
    validation_set = np.array(f["volumes/masks/validation"])
    validation_mask[validation_set == 1] = 1
    validation_mask[validation_set == 0xFFFFFFFFFFFFFFFF] = 1
    # sanity_check = np.ones_like(mask)
    # sanity_check[]==0
    # clefts_orig[validation_mask]=0
    assert clefts_outside == np.sum(validation_set == 0xFFFFFFFFFFFFFFFF - 2)
    g.create_dataset(
        "volumes/masks/validation",
        data=validation_mask,
        dtype="float32",
        chunks=(26, 256, 256),
    )
    for k, v in f["volumes/masks/validation"].attrs.items():
        g["volumes/masks/validation"].attrs.create(k, v)
    del validation_mask
    del validation_set
    g.create_dataset(
        "volumes/raw", data=np.array(f["volumes/raw"]), chunks=(26, 256, 256)
    )
    for k, v in f["volumes/raw"].attrs.items():
        g["volumes/raw"].attrs.create(k, v)
    # print(np.sum(clefts_orig == 0xffffffffffffffff-2))
    # print(clefts_orig.shape)
    # print(np.unique(clefts_orig))
    # raise Exception

    for k, v in f.attrs.items():
        g.attrs.create(k, v)

    f.close()
    g.close()


if __name__ == "__main__":
    for sample in ["A", "B", "C"]:  # , 'B', 'C']:
        rewrite_mask(sample)
