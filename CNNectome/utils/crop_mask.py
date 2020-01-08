import zarr
import numcodecs
import itertools
import numpy as np


def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in list(itertools.combinations(range(N), N - 1))[::-1]:
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def crop_mask(
    filename,
    mask_ds="volumes/masks/training",
    target_ds="volumes/masks/training_cropped",
    padding=256,
):
    print(
        "cropping {0:} of {1:} to {2:} with padding {3:}".format(
            mask_ds, filename, target_ds, padding
        )
    )
    # open file
    zf = zarr.open(filename, mode="a")
    # read mask
    ds = zf[mask_ds]
    mask = ds[:]
    # find bounding box of mask
    bb = bbox_ND(mask)
    sl = (
        slice(bb[0] - padding, bb[1] + 1 + padding),
        slice(bb[2] - padding, bb[3] + 1 + padding),
        slice(bb[4] - padding, bb[5] + 1 + padding),
    )
    off = tuple(
        np.array([sl[0].start, sl[1].start, sl[2].start])
        * np.array(ds.attrs["resolution"])
    )
    print("bounding box", sl)
    print("offset", off)
    # crop
    mask_cropped = mask[sl]
    print("shape", mask_cropped.shape)
    # save
    zf.empty(
        name=target_ds,
        shape=mask_cropped.shape,
        compressor=numcodecs.GZip(6),
        dtype=mask_cropped.dtype,
        chunks=ds.chunks,
    )
    zf[target_ds][:] = mask_cropped
    # save offset
    zf[target_ds].attrs["offset"] = off[::-1]
    zf[target_ds].attrs["resolution"] = ds.attrs["resolution"]


if __name__ == "__main__":
    dataset_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v061719_o750x750x750_m1800x1800x1800_8nm/{0:}"
    list_of_ds = [
        "crop1.n5",
        "crop3.n5",
        "crop4.n5",
        "crop6.n5",
        "crop7.n5",
        "crop8.n5",
        "crop9.n5",
        "crop13.n5",
        "crop14.n5",
        "crop15.n5",
        "crop16.n5",
        "crop18.n5",
        "crop19.n5",
        "crop20.n5",
        "crop21.n5",
        "crop22.n5",
        "crop31.n5",
        "crop33.n5",
        "crop34.n5",
    ]
    padding = 256
    for ds in list_of_ds:
        crop_mask(
            dataset_dir.format(ds),
            target_ds="volumes/masks/training_off{0:}".format(padding),
            padding=padding,
        )
    # crop_mask(dataset_dir.format('crop1.n5'))
