import numpy as np
import numcodecs
import zarr
import itertools
import argparse


def get_slice_shape(sl):
    sh = tuple(
        [sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, sl[2].stop - sl[2].start]
    )
    return sh


def save_mask(data_file, data_dsname_raw, data_dsname_mask, chunks, cut_axis):
    f = zarr.open(data_file, "a")

    raw_ds = f[data_dsname_raw]
    mask_ds = f.empty(
        name=data_dsname_mask,
        shape=raw_ds.shape,
        chunks=(256, 256, 256),
        dtype=np.uint64,
        compressor=numcodecs.GZip(6),
    )
    mask_ds.attrs["pixelResolution"] = raw_ds.attrs["pixelResolution"]
    start = (0, 0, 0)
    end = mask_ds.shape
    boundary = int(0.5 * mask_ds.shape[cut_axis])
    for z, y, x in itertools.product(
        range(start[0], end[0], chunks[0]),
        range(start[1], end[1], chunks[1]),
        range(start[2], end[2], chunks[2]),
    ):
        sl = (
            slice(z, min(z + chunks[0], end[0])),
            slice(y, min(y + chunks[1], end[1])),
            slice(x, min(x + chunks[2], end[2])),
        )

        if sl[cut_axis].stop <= boundary:
            mask_ds[sl] = np.ones(get_slice_shape(sl), dtype=np.uint64)
        elif boundary <= sl[cut_axis].start:
            mask_ds[sl] = np.zeros(get_slice_shape(sl), dtype=np.uint64)
        else:
            sl_before = tuple(
                slice(sl[k].start, sl[k].stop, sl[k].step)
                if k != cut_axis
                else slice(sl[k].start, boundary, sl[k].step)
                for k in range(3)
            )
            mask_ds[sl_before] = np.ones(get_slice_shape(sl_before), dtype=np.uint64)
            sl_after = tuple(
                slice(sl[k].start, sl[k].stop, sl[k].step)
                if k != cut_axis
                else slice(boundary, sl[k].stop, sl[k].step)
                for k in range(3)
            )
            mask_ds[sl_after] = np.zeros(get_slice_shape(sl_after), dtype=np.uint64)


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("n5file", type=str)
    parser.add_argument("--mask_dataset", type=str, default="volumes/masks/training")
    parser.add_argument("--raw_dataset", type=str, default="volumes/raw/s0")
    parser.add_argument("--cut_axis", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=(256, 256, 256), nargs="+")
    args = parser.parse_args()
    save_mask(
        args.n5file,
        args.raw_dataset,
        args.mask_dataset,
        tuple(args.chunk_size),
        args.cut_axis,
    )


if __name__ == "__main__":
    main()
