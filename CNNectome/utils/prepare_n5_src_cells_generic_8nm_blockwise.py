import zarr
import numcodecs
import h5py
import numpy as np
import collections
import logging
import skimage.transform
import multiprocessing
import itertools
import argparse


def multiple_inputs(func):
    def wrapper(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            return func(data, *args, **kwargs)
        else:
            return [func(d, *args, **kwargs) for d in data]

    return wrapper


def add_ds(target, name, shape, dtype, chunks, resolution, offset, **kwargs):
    logging.info("Preparing dataset {0:} in {1:}".format(name, target.path))
    ds = target.empty(
        name=name, shape=shape, chunks=chunks, dtype=dtype, compressor=numcodecs.GZip(6)
    )
    ds.attrs["resolution"] = resolution
    ds.attrs["offset"] = offset
    for k in kwargs:
        ds.attrs[k] = kwargs[k]
    return ds


def blockwise_process_and_write(func):
    def wrapper(target_ds, data, *args, **kwargs):
        if "chunks" not in kwargs:
            chunks = target_ds.chunks
        else:
            chunks = kwargs["chunks"]
        if "start" not in kwargs:
            start = (0, 0, 0)
        else:
            start = kwargs["start"]
        if "end" not in kwargs:
            end = target_ds.shape
        else:
            end = kwargs["end"]
        if "offset" not in kwargs:
            offset = (0, 0, 0)
        else:
            offset = tuple(kwargs["offset"])
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
            if offset == (0, 0, 0):
                sl_src = sl
            else:
                sl_src = (
                    slice(
                        z - offset[0],
                        min(z + chunks[0] - offset[0], end[0] - offset[0]),
                    ),
                    slice(
                        y - offset[1],
                        min(y + chunks[1] - offset[1], end[1] - offset[1]),
                    ),
                    slice(
                        x - offset[2],
                        min(x + chunks[2] - offset[2], end[2] - offset[2]),
                    ),
                )
            target_ds[sl] = func(target_ds, np.array(data[sl_src]), *args, **kwargs)

    return wrapper


def initialize_blockwise(target_ds, value, *args, **kwargs):
    if "chunks" not in kwargs:
        chunks = target_ds.chunks
    else:
        chunks = kwargs["chunks"]
    if "start" not in kwargs:
        start = (0, 0, 0)
    else:
        start = kwargs["start"]
    if "end" not in kwargs:
        end = target_ds.shape
    else:
        end = kwargs["end"]
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
        size = (
            sl[0].stop - sl[0].start,
            sl[1].stop - sl[1].start,
            sl[2].stop - sl[2].start,
        )
        target_ds[sl] = np.ones(size, dtype=target_ds.dtype) * value


def histogram_blockwise(data, bins, *args, **kwargs):
    if "chunks" not in kwargs:
        chunks = data.chunks
    else:
        chunks = kwargs["chunks"]
    if "start" not in kwargs:
        start = (0, 0, 0)
    else:
        start = kwargs["start"]
    if "end" not in kwargs:
        end = data.shape
    else:
        end = kwargs["end"]
    hist = np.zeros(len(bins) - 1)
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
        hist = hist + np.histogram(np.array(data[sl]), bins=bins)[0]
    return hist, bins


def multiply_blockwise(target_ds, data1, data2, *args, **kwargs):
    if "chunks" not in kwargs:
        chunks = target_ds.chunks
    else:
        chunks = kwargs["chunks"]
    if "start" not in kwargs:
        start = (0, 0, 0)
    else:
        start = kwargs["start"]
    if "end" not in kwargs:
        end = target_ds.shape
    else:
        end = kwargs["end"]
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
        target_ds[sl] = data1[sl] * data2[sl]


def generate_mask_blockwise(target_ds, data, factor, bg_label, **kwargs):
    if not isinstance(factor, tuple):
        factor = (factor,) * len(data.shape)
    if "chunks" not in kwargs:
        chunks = target_ds.chunks
    else:
        chunks = kwargs["chunks"]
    if "start" not in kwargs:
        start = (0, 0, 0)
    if "end" not in kwargs:
        end = target_ds.shape
    for z, y, x in itertools.product(
        range(start[0], end[0], chunks[0]),
        range(start[1], end[1], chunks[1]),
        range(start[2], end[2], chunks[2]),
    ):
        factor = np.array(factor).astype(np.int)
        src_sl = (
            slice(z * factor[0], min((z + chunks[0]) * factor[0], end[0] * factor[0])),
            slice(y * factor[0], min((y + chunks[1]) * factor[1], end[1] * factor[1])),
            slice(x * factor[0], min((x + chunks[2]) * factor[2], end[2] * factor[2])),
        )
        down_sl = (
            slice(None, None, factor[0]),
            slice(None, None, factor[1]),
            slice(None, None, factor[2]),
        )
        sl = (
            slice(z, min(z + chunks[0], end[0])),
            slice(y, min(y + chunks[1], end[1])),
            slice(x, min(x + chunks[2], end[2])),
        )

        target_ds[sl] = (np.array(data[src_sl][down_sl]) != bg_label).astype(np.uint64)


@blockwise_process_and_write
def map(target_ds, data, mapping, *args, **kwargs):
    return mapping[data]


@blockwise_process_and_write
def copy_ds(target_ds, data, *args, **kwargs):
    return data


def downsample_copy_ds(target_ds, data, factor, *args, **kwargs):
    if not isinstance(factor, tuple):
        factor = (factor,) * len(data.shape)
    if "chunks" not in kwargs:
        chunks = target_ds.chunks
    else:
        chunks = kwargs["chunks"]
    if "start" not in kwargs:
        start = (0, 0, 0)
    else:
        start = kwargs["start"]
    if "end" not in kwargs:
        end = target_ds.shape
    else:
        end = kwargs["end"]
    if "offset" not in kwargs:
        offset = (0, 0, 0)
    else:
        offset = tuple(kwargs["offset"])
    factor = np.array(factor).astype(np.int)
    for z, y, x in itertools.product(
        range(start[0], end[0], chunks[0]),
        range(start[1], end[1], chunks[1]),
        range(start[2], end[2], chunks[2]),
    ):
        factor = np.array(factor).astype(np.int)
        src_sl = (
            slice(
                z * factor[0] - offset[0],
                min(
                    (z + chunks[0]) * factor[0] - offset[0],
                    end[0] * factor[0] - offset[0],
                ),
                factor[0],
            ),
            slice(
                y * factor[1] - offset[1],
                min(
                    (y + chunks[1]) * factor[1] - offset[1],
                    end[1] * factor[1] - offset[1],
                ),
                factor[1],
            ),
            slice(
                x * factor[2] - offset[2],
                min(
                    (x + chunks[2]) * factor[2] - offset[2],
                    end[2] * factor[2] - offset[2],
                ),
                factor[2],
            ),
        )
        sl = (
            slice(z - offset[0], min(z + chunks[0] - offset[0], end[0] - offset[0])),
            slice(y - offset[1], min(y + chunks[1] - offset[1], end[1] - offset[1])),
            slice(x - offset[2], min(x + chunks[2] - offset[2], end[2] - offset[2])),
        )
        print("target", sl, "source", src_sl)
        x = (np.array(data[src_sl])).astype(np.uint64)
        print(x.shape)
        print(target_ds[sl].shape)
        target_ds[sl] = x


@blockwise_process_and_write
def contrast_adaptation(target_ds, data, min_ad, max_ad, *args, **kwargs):
    scale = 255.0 / (float(max_ad) - float(min_ad))
    shift = -scale * float(min_ad)
    return np.clip((data * scale + shift).round(), 0, 255).astype(np.uint8)


def main_multiscale(
    orig,
    target,
    labelnames,
    mapping,
    min_ad,
    max_ad,
    src_label_name="volumes/labels/gt",
    specified_masks=None,
    separate_datasets=None,
):
    if specified_masks is None:
        specified_masks = dict()
    if separate_datasets is None:
        separate_datasets = dict()
    if "volumes" not in target.keys():
        target.create_group("volumes")
    if "labels" not in target["volumes"]:
        target["volumes"].create_group("labels")
    if "masks" not in target["volumes"]:
        target["volumes"].create_group("masks")

    # raw datset
    raw = orig["volumes/raw"]
    bg_label = 18446744073709551613
    label_maxid = 36
    logging.info(
        "RAW dataset {0:} has resolution {1:} and offset {2:}".format(
            raw.shape, raw.attrs["resolution"], raw.attrs["offset"]
        )
    )
    raw_ds = add_ds(
        target,
        "volumes/raw",
        raw.shape,
        raw.dtype,
        raw.chunks,
        list(raw.attrs["resolution"]),
        [0.0, 0.0, 0.0],
    )
    contrast_adaptation(
        raw_ds, raw, min_ad, max_ad, chunks=tuple(np.array(raw_ds.chunks) * 5)
    )
    for subsample_variant in range(8):
        raw_ss = orig["volumes/subsampled/raw/{0:}".format(subsample_variant)]
        raw_ss_ds = add_ds(
            target,
            "volumes/subsampled/raw/{0:}".format(subsample_variant),
            raw_ss.shape,
            raw_ss.dtype,
            raw_ss.chunks,
            list(raw_ss.attrs["resolution"]),
            [0.0, 0.0, 0.0],
        )
        contrast_adaptation(
            raw_ss_ds,
            raw_ss,
            min_ad,
            max_ad,
            chunks=tuple(np.array(raw_ss_ds.chunks) * 5),
        )

    # labels
    labels = orig[src_label_name]
    logging.info(
        "LABELS dataset {0:} has resolution {1:} and offset {2:}".format(
            labels.shape, labels.attrs["resolution"], labels.attrs["offset"]
        )
    )
    hist = histogram_blockwise(labels, bins=list(range(label_maxid + 1)))
    cont = [i for i, count in zip(hist[1], hist[0]) if count > 0]
    labels_ds = add_ds(
        target,
        "volumes/labels/all",
        tuple([rs * 2 for rs in raw.shape]),
        np.uint64,
        labels.chunks,
        list(labels.attrs["resolution"]),
        [0.0, 0.0, 0.0],
        orig_ids=cont,
        orig_counts=list(hist[0]),
        mapping=mapping,
    )
    labels_ds = target["volumes/labels/all"]
    padding_before = (
        (
            (
                np.array(labels.attrs["offset"][::-1])
                - np.array(labels.attrs["resolution"][::-1]) / 2.0
            )
            + np.array(raw.attrs["resolution"][::-1] / 2.0)
        )
        / np.array(labels.attrs["resolution"][::-1])
    ).astype(np.int)
    padding_after = (
        2 * np.array(raw.shape) - padding_before - np.array(labels.shape)
    ).astype(np.int)
    initialize_blockwise(
        labels_ds, bg_label, chunks=tuple(np.array(labels_ds.chunks) * 5)
    )
    if mapping is None:
        copy_ds(
            labels_ds,
            labels,
            start=padding_before,
            end=labels_ds.shape - padding_after,
            offset=padding_before,
            chunks=tuple(np.array(labels_ds.chunks) * 5),
        )
        relabeled_cont = cont
        relabeled_hist = hist

    else:
        map(
            labels_ds,
            labels,
            mapping,
            start=padding_before,
            end=labels_ds.shape - padding_after,
            offset=padding_before,
            chunks=tuple(np.array(labels_ds.chunks) * 5),
        )
        relabeled_hist = histogram_blockwise(labels_ds, bins=range(label_maxid + 1))
        relabeled_cont = [
            i for i, count in zip(relabeled_hist[1], relabeled_hist[0]) if count > 0
        ]
    labels_ds.attrs["relabeled_ids"] = relabeled_cont
    labels_ds.attrs["relabeled_counts"] = list(relabeled_hist[0])

    # training mask
    mask_ds = add_ds(
        target,
        "volumes/masks/training",
        raw.shape,
        np.uint64,
        tuple(labels.chunks),
        tuple(raw.attrs["resolution"]),
        [0.0, 0.0, 0.0],
    )
    generate_mask_blockwise(
        mask_ds,
        labels_ds,
        (2.0, 2.0, 2.0),
        bg_label,
        chunks=tuple(np.array(mask_ds.chunks) * 5),
    )

    for l in labelnames:
        label_mask_ds = add_ds(
            target,
            "volumes/masks/" + l,
            mask_ds.shape,
            mask_ds.dtype,
            mask_ds.chunks,
            mask_ds.attrs["resolution"],
            mask_ds.attrs["offset"],
        )
        if l not in specified_masks.keys():
            copy_ds(
                label_mask_ds, mask_ds, chunks=tuple(np.array(label_mask_ds.chunks) * 5)
            )
        else:
            if specified_masks[l] == 0:
                initialize_blockwise(
                    label_mask_ds, 0, chunks=tuple(np.array(label_mask_ds.chunks) * 5)
                )
            elif specified_masks[l] == 1:
                initialize_blockwise(
                    label_mask_ds, 1, chunks=tuple(np.array(label_mask_ds.chunks) * 5)
                )
            elif isinstance(specified_masks[l], str):
                # assert orig[specified_masks[l]].shape == mask_ds.shape
                initialize_blockwise(
                    label_mask_ds, 0, chunks=tuple(np.array(mask_ds.chunks) * 5)
                )
                downsample_copy_ds(
                    label_mask_ds,
                    orig[specified_masks[l]],
                    (2.0, 2.0, 2.0),
                    start=(padding_before / 2.0).astype(np.int),
                    end=(mask_ds.shape - padding_after / 2.0).astype(np.int),
                    offset=padding_before.astype(np.int),
                    chunks=tuple(np.array(mask_ds.chunks) * 5),
                )
                multiply_blockwise(
                    label_mask_ds,
                    label_mask_ds,
                    mask_ds,
                    chunks=tuple(np.array(label_mask_ds.chunks) * 5),
                )
            else:
                raise ValueError

    for l in labelnames:
        if l in separate_datasets.keys():
            label_orig = separate_datasets[l]
            label_data = label_orig["volumes/labels/" + l]
            logging.info(
                "{0:} dataset {1:} has resolution {2:} and offset {3:}".format(
                    l.upper(),
                    label_data.shape,
                    label_data.attrs["resolution"],
                    label_data.attrs["offset"],
                )
            )
            label_ds = add_ds(
                target,
                "volumes/labels/" + l,
                labels_ds.shape,
                np.uint64,
                labels_ds.chunks,
                list(labels_ds.attrs["resolution"]),
                list(labels_ds.attrs["offset"]),
            )
            initialize_blockwise(
                label_ds, bg_label, chunks=tuple(np.array(label_ds.chunks) * 5)
            )
            copy_ds(
                label_ds,
                label_data,
                start=padding_before,
                end=label_ds.shape - padding_after,
                offset=padding_before,
                chunks=tuple(np.array(label_ds.chunks) * 5),
            )


def generate_integral_mask(
    crop, offset, mask_ds_name="volumes/masks/training", datestr="061719"
):
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/{2:}.n5".format(
            datestr, offset, crop
        ), mode="a"
    )
    mask_ds = target[mask_ds_name]
    target_ds_name = mask_ds_name + "_integral"
    integral_mask_ds = add_ds(
        target,
        target_ds_name,
        mask_ds.shape,
        np.uint64,
        (8, 8, 8),
        list(mask_ds.attrs["resolution"]),
        list(mask_ds.attrs["offset"]),
    )
    logging.info("Computing integral mask...")
    integral_mask_ds[:] = skimage.transform.integral_image(mask_ds[:])


def main_multiscale_crop1(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop1_Periphery"
        "/Cell2_Crop1_2000x2000x1800+5716-525-250_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop1.n5".format(
            datestr, offset
        ),
        mode="a"
    )
    # mapping = np.array([0, 4, 3, 10, 16, 2, 1, 1, 17, 11, 8, 30, 18, 19, 35, 9])
    # [0, mito lumen, mito membrane, MVB membrane, er membrane, plasma membrane, ECS, ECS, er lumen, MVB lumen,
    # vesicle membrane, microtubules, ERES membrane, ERES lumen, cytosol, vesicle lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop3(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop3_Mitos"
        "/HeLa_Cell2_Crop3_1900x1900x1800+7050-550+3224_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop3.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 10, 2, 3, 8, 30, 17, 4, 11, 9, 18, 35, 1, 5, 19, 12, 13])
    # [0, er membrane, MVB membrane, plasma membrane, mito membrane, vesicle membrane, microtubules, er lumen,
    # mito lumen, MVB lumen, vesicle lumen, ERES membrane, cytosol, ECS, mito DNA, ERES lumen, lysosome membrane,
    # lysososme lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop4(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop4_Centrosome"
        "/HeLa_Cell2_Crop4_1800x1800x1800+5350-550+1956_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop4.n5".format(
            datestr, offset
        ),
        mode="a"
    )
    # mapping = np.array([0, 20, 24, 10, 30, 28, 22, 23, 6, 35, 16, 7, 13, 11, 17, 21, 33, 32, 8, 9, 12])
    # [0, NE membrane, HChrom , MVB membrane, microtubules, nucleoplasm, nuclear pore outside, nuclear pore inside,
    # golgi membrane, cytosol, er membrane, golgi lumen, lysosome lumen, MVB lumen, er lumen, NE lumen,
    # subidstal appendages, distal appendages, vesicle membrane, vesicle lumen, lysosome membrane]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig,
        target,
        labels,
        mapping,
        min_ad,
        max_ad,
        specified_masks={
            "ribosomes": 0,
            "NHChrom": 1,
            "EChrom": 1,
            "NEChrom": 1,
            "microtubules": "volumes/masks/centrosomes",
            "centrosomes": "volumes/masks/centrosomes",
        },
        separate_datasets={"centrosomes": orig},
    )


def main_multiscale_crop6(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop6_Ribosome"
        "/Cell2_Crop6_1800x1800x1800+2245-375+2224_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop6.n5".format(
            datestr, offset
        ),
        mode="a"
    )
    # mapping = np.array([30, 16, 17, 3, 4, 10, 11, 35, 18, 19, 30])
    # [0, er membrane, er lumen, mito membrane, mito lumen, MVB membrane, MVB lumen, cytosol, ERES membrane,
    # ERES lumen, microtubules]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop7(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop7_PM"
        "/Cell2_Crop7_1800x1800x1800+7530-510-860_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop7.n5".format(
            datestr, offset
        ),
        mode="a"
    )
    # mapping = np.array([0, 2, 2, 1, 35, 35, 30, 8, 9, 3, 4])
    # [0, plasma membrane, plasma membrane, ECS, cytosol, cytosol, microtubules, vesicle membrane, vesicle lumen,
    # mito membrane, mito lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop8(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop8_ERES001"
        "/Cell2_Crop8_1800x1800x1800+2585-520+399_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop8.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 18, 19, 30, 35, 17, 8, 9, 10])
    # [0, er membrane, ERES membrane, ERES lumen, microtubules, cytosol, er lumen, vesicle membrane, vesicle lumen,
    # MVB membrane]
    mapping = None
    max_ad = 207.0
    min_ad = 89.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop9(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop9_ERES002"
        "/HeLa_Cell2_Crop9_1800x1800x1801+2050-430+735_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop9.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 18, 3, 4, 35, 16, 17, 19, 10, 11, 30, 8, 9])
    # [0, ERES membrane, mito membrane, mito lumen, cytosol, er membrane, er lumen, ERES lumen, MVB membrane,
    # MVB lumen, microtubules, vesicle membrane, vesicle lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop13(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop13_ERES006"
        "/HeLa_Cell2_Crop13_1800x1800x1800+4340-60+3087_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop13.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 18, 19, 10, 11, 35, 30, 20, 21, 36, 28, 17, 8, 9])
    # [0, er membrane, ERES membrane, ERES lumen, MVB  membrane, MVB lumen, cytosol, microtubules, nuclear envelope
    # membrane, nuclear envelope lumen, chromatin, nucleoplasm, er lumen, vesicle membrane, vesicle lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop14(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop14_ERES007"
        "/Cell2_Crop14_1800x1800x1801+5505-450+3548_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop14.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 19, 18, 30, 35, 17, 8, 9])
    # [0, er membrane, ERES lumen, ERES membrane, microtubules, cytosol, er lumen, vesicle membrane, vesicle lumen]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop15(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop15_ERES008"
        "/Cell2_Crop15_1800x1800x1800+5305-380+3542_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop15.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 17, 18, 19, 35, 30])
    # [0, er membrane, er lumen, ERES membrane, ERES lumen, cytosol, microtubules]
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop16(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop16_Nucleus001"
        "/HeLa_Cell2_Crop16_1800x1800x1800+4410+310+1100_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop16.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    mapping = None
    min_ad = 89.0
    max_ad = 207.0
    main_multiscale(
        orig,
        target,
        labels,
        mapping,
        min_ad,
        max_ad,
        specified_masks={"ribosomes": 1},
        separate_datasets={"nucleolus": orig, "ribosomes": orig},
    )


def main_multiscale_crop18(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop18_MVB"
        "/Cell2_Crop18_1800x1800x1800+1110-600+2885_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop18.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 30, 35, 16, 17, 10, 11, 8, 9])
    # [0, microtubules, cytosol, er membrane, er lumen, MVB membrane, MVB lumen, vesicle membrane, vesicle lumen]
    mapping = None
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop19(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell2_Crop19_BadLD"
        "/Cell2_Crop19_1800x1800x1801+6075-425+4062_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop19.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 16, 17, 14, 15, 30, 8, 9, 35])
    # [0, er membrane, er lumen, LD membrane, LD lumen, microtubules, vesicle membrane, vesicle lumen, cytosol]
    mapping = None
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop20(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/SUM159_Cell2_Crop20_LD001"
        "/Cell2_Crop20_1800x1800x1801+3905-55+4643_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop20.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 3, 4, 14, 15, 16, 17, 10, 11, 35])
    # [0, mito membrane, mito lumen, LD membrane, LD lumen, er membrane, er lumen, MVB membrane, MVB lumen, cytosol]
    mapping = None
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop21(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/SUM159_Cell2_Crop21_LD002"
        "/Cell2_Crop21_1800x1800x1801+3740-190+4703_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop21.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 3, 4, 16, 17, 14, 15, 35])
    # [0, mito membrane, mito lumen, er membrane, er lumen, LD membrane LD lumen, cytosol]
    mapping = None
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop22(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/SUM159_Cell2_Crop22_LD003"
        "/Cell2_Crop22_1800x1800x1800+3195-355+4530_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop22.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    # mapping = np.array([0, 35, 16, 17, 14, 15, 10, 11, 30])
    # [0, cytosol, er membrane, er lumen, LD membrane, LD lumen, MVB membrane, MVB lumen, microtubules]
    mapping = None
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop31(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/Mac_Cell2_Crop31_Mito001"
        "/Mac_Cell2_Crop31_1800x1800x1800+3195+0+8005_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop31.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    mapping = None
    min_ad = 0.75 * 255.0
    max_ad = 255.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop33(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell3_Crop33_Mito001"
        "/HeLa_Cell3_Crop33_1800x1800x1800+2650-570+4850_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop33.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    mapping = None
    min_ad = 0.0
    max_ad = 255.0 * 1.1
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop34(labels, offset, datestr="061719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations061719/HeLa_Cell3_Crop34_Mito002"
        "/HeLa_Cell3_Crop34_1800x1800x1800+5870-150+5100_8nm.h5",
        "r",
    )
    target = zarr.open(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop34.n5".format(
            datestr, offset
        ),
        mode="a",
    )
    mapping = None
    min_ad = 0.0
    max_ad = 255.0 * 1.1
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def run_main_multiscale():
    logging.basicConfig(level=logging.INFO)
    labels = [  #'ecs',
        # 'plasma_membrane',
        # 'mito_membrane',
        # 'mito',
        # 'mito_DNA',
        # 'golgi_membrane',
        # 'golgi',
        # 'vesicle_membrane',
        # 'vesicle',
        # 'MVB_membrane',
        # 'MVB',
        # 'lysosome_membrane',
        # 'lysosome',
        # 'LD_membrane',
        # 'LD',
        # 'er_membrane',
        # 'er',
        # 'ERES',
        # 'NE',
        # 'nuclear_pore',
        # 'nuclear_pore_in',
        # 'chromatin',
        # 'NHChrom',
        # 'EChrom',
        # 'NEChrom',
        # 'nucleus',
        # 'nucleolus',
        # 'microtubules',
        # 'centrosome',
        # 'distal_app',
        # 'subdistal_app',
        # 'ribosomes'
    ]
    offset = "o750x750x750_m1800x1800x1800"
    main_multiscale_crop1(labels, offset)
    main_multiscale_crop7(labels, offset)
    main_multiscale_crop14(labels, offset)
    main_multiscale_crop22(labels, offset)
    main_multiscale_crop8(labels, offset)
    main_multiscale_crop19(labels, offset)
    main_multiscale_crop9(labels, offset)
    main_multiscale_crop13(labels, offset)
    main_multiscale_crop15(labels, offset)
    main_multiscale_crop18(labels, offset)
    main_multiscale_crop20(labels, offset)
    main_multiscale_crop21(labels, offset)

    main_multiscale_crop3(labels, offset)
    main_multiscale_crop4(labels, offset)
    main_multiscale_crop6(labels, offset)


def main():
    logging.basicConfig(level=logging.INFO)
    labels = [
        "ecs",
        "plasma_membrane",
        "mito_membrane",
        "mito",
        "mito_DNA",
        "golgi_membrane",
        "golgi",
        "vesicle_membrane",
        "vesicle",
        "MVB_membrane",
        "MVB",
        "lysosome_membrane",
        "lysosome",
        "LD_membrane",
        "LD",
        "er_membrane",
        "er",
        "ERES",
        "NE",
        "nuclear_pore",
        "nuclear_pore_in",
        "chromatin",
        "NHChrom",
        "EChrom",
        "NEChrom",
        "nucleus",
        "nucleolus",
        "microtubules",
        "centrosomes",
        "distal_app",
        "subdistal_app",
        "ribosomes",
    ]
    offset = "o750x750x750_m1800x1800x1800"
    parser = argparse.ArgumentParser(
        description="Generate n5 training files from annotations"
    )
    parser.add_argument(
        "crop",
        type=str,
        help="crop that should be processed, e.g. crop1",
        choices=[
            "crop1",
            "crop3",
            "crop4",
            "crop6",
            "crop7",
            "crop8",
            "crop9",
            "crop13",
            "crop14",
            "crop15",
            "crop16",
            "crop18",
            "crop19",
            "crop20",
            "crop21",
            "crop22",
            "crop31",
            "crop33",
            "crop34",
        ],
    )
    parser.add_argument("-integral_mask", action="store_true")
    parser.add_argument(
        "mask_ds",
        type=str,
        help="specify dataset of mask for computing integral mask",
        default="volumes/masks/training",
    )
    args = parser.parse_args()
    print(args.mask_ds)
    if args.integral_mask:
        generate_integral_mask(args.crop, offset, args.mask_ds)
    else:
        if args.crop == "crop1":
            main_multiscale_crop1(labels, offset)
        elif args.crop == "crop3":
            main_multiscale_crop3(labels, offset)
        elif args.crop == "crop4":
            main_multiscale_crop4(labels, offset)
        elif args.crop == "crop6":
            main_multiscale_crop6(labels, offset)
        elif args.crop == "crop7":
            main_multiscale_crop7(labels, offset)
        elif args.crop == "crop8":
            main_multiscale_crop8(labels, offset)
        elif args.crop == "crop9":
            main_multiscale_crop9(labels, offset)
        elif args.crop == "crop13":
            main_multiscale_crop13(labels, offset)
        elif args.crop == "crop14":
            main_multiscale_crop14(labels, offset)
        elif args.crop == "crop15":
            main_multiscale_crop15(labels, offset)
        elif args.crop == "crop16":
            main_multiscale_crop16(labels, offset)
        elif args.crop == "crop18":
            main_multiscale_crop18(labels, offset)
        elif args.crop == "crop19":
            main_multiscale_crop19(labels, offset)
        elif args.crop == "crop20":
            main_multiscale_crop20(labels, offset)
        elif args.crop == "crop21":
            main_multiscale_crop21(labels, offset)
        elif args.crop == "crop22":
            main_multiscale_crop22(labels, offset)
        elif args.crop == "crop31":
            main_multiscale_crop31(labels, offset)
        elif args.crop == "crop33":
            main_multiscale_crop33(labels, offset)
        elif args.crop == "crop34":
            main_multiscale_crop34(labels, offset)
        else:
            raise ValueError("Unknown argument for crop: {0:}".format(args.crop))


if __name__ == "__main__":
    main()
