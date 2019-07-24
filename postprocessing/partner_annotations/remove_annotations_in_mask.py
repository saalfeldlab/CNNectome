from cremi.io import CremiFile
import h5py
import shutil
import numpy as np
import sys
import z5py


def remove_annotations_in_mask(filename, mask_filename, mask_ds):
    fh = CremiFile(filename, "a")
    if mask_filename.endswith(".h5") or mask_filename.endswith(".hdf"):
        maskfh = h5py.File(mask_filename, "r")
    else:
        maskfh = z5py.File(mask_filename, use_zarr_format=False)
    mask = maskfh[mask_ds]
    off = mask.attrs["offset"]
    res = mask.attrs["resolution"]
    mask = mask[:]
    ann = fh.read_annotations()
    shift = sub(ann.offset, off)

    ids = ann.ids()
    rmids = []
    for i in ids:
        t, loc = ann.get_annotation(i)
        vx_idx = (np.array(add(loc, shift)) / res).astype(np.int)
        if not mask[tuple(vx_idx)]:
            rmids.append(i)
    for i in rmids:
        print("removing {0:}".format(i))
        ann.remove_annotation(i)
    fh.write_annotations(ann)


def sub(a, b):
    return tuple([a[d] - b[d] for d in range(len(b))])


def add(a, b):
    return tuple([a[d] + b[d] for d in range(len(b))])


def main(filename, mask_fn):
    data_name = filename.split("/")[-1]
    if "A" in data_name and "B" not in data_name and "C" not in data_name:
        sample = "A"
    elif "A" not in data_name and "B" in data_name and "C" not in data_name:
        sample = "B"
    elif "A" not in data_name and "B" not in data_name and "C" in data_name:
        sample = "C"
    else:
        sample = data_name[0]

    # mask_fn = '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{
    # 0:}_padded_20170424.aligned.0bg.hdf'.format(
    #    sample)
    mask_dss = ["volumes/masks/training", "volumes/masks/validation"]
    assert filename.endswith("hdf") or filename.endswith(".h5")
    shutil.copy(filename, filename.replace(".hdf", ".training.hdf"))
    shutil.copy(filename, filename.replace(".hdf", ".validation.hdf"))
    tgts = [
        filename.replace(".hdf", ".training.hdf"),
        filename.replace(".hdf", ".validation.hdf"),
    ]
    for mask_ds, fn in zip(mask_dss, tgts):
        remove_annotations_in_mask(fn, mask_fn, mask_ds)


if __name__ == "__main__":
    f = sys.argv[1]
    mf = sys.argv[2]
    main(f, mf)
