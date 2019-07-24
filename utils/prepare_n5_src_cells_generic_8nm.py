from __future__ import print_function
import z5py
import h5py
import numpy as np
import collections
import datetime
import logging
import skimage.transform


def add_ds(target, name, data, chunks, resolution, offset, **kwargs):
    if name not in target:
        logging.info("Writing dataset {0:} to {1:}".format(name, target.path))
        ds = target.create_dataset(
            name,
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            compression="gzip",
            type="gzip",
            level=6,
        )
        target[name].attrs["resolution"] = resolution
        target[name].attrs["offset"] = offset
        target[name][:] = np.array(data)
        for k in kwargs:
            target[name].attrs[k] = kwargs[k]
    else:
        logging.info(
            "Dataset {0:} already exists in {1:}, not overwriting".format(
                name, target.path
            )
        )
        ds = target[name]
    return ds


def add_subset_label_ds(target, labels, name, label_ids, chunks, resolution):
    if not isinstance(label_ids, collections.Iterable):
        label_ids = (label_ids,)
    add_ds(
        target,
        name,
        np.logical_or.reduce([labels == lid for lid in label_ids]).astype(labels.dtype),
        chunks,
        resolution,
        [0.0, 0.0, 0.0],
    )


def contrast_adaptation(raw, min_ad, max_ad):
    scale = 255.0 / (float(max_ad) - float(min_ad))
    shift = -scale * float(min_ad)
    return (raw * scale + shift).round().astype(np.uint8)


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

    # raw dataset
    raw = orig["volumes/raw"]
    logging.info(
        "RAW dataset {0:} has resolution {1:} and offset {2:}".format(
            raw.shape, raw.attrs["resolution"], raw.attrs["offset"]
        )
    )

    add_ds(
        target,
        "volumes/raw",
        contrast_adaptation(np.array(raw), min_ad, max_ad),
        raw.chunks,
        list(raw.attrs["resolution"]),
        [0.0, 0.0, 0.0],
    )
    for subsample_variant in range(8):
        raw_ss = orig["volumes/subsampled/raw/{0:}".format(subsample_variant)]
        if "volumes/subsampled/raw/{0:}".format(subsample_variant) not in target:
            add_ds(
                target,
                "volumes/subsampled/raw/{0:}".format(subsample_variant),
                contrast_adaptation(np.array(raw_ss), min_ad, max_ad),
                raw_ss.chunks,
                list(raw_ss.attrs["resolution"]),
                [0.0, 0.0, 0.0],
            )
        else:
            logging.info(
                "Dataset {0:} already exists in {1:}, not overwriting".format(
                    "volumes/subsampled/raw/{" "0:}".format(subsample_variant),
                    target.path,
                )
            )
    del raw_ss

    # generic labels
    labels = orig[src_label_name]
    logging.info(
        "LABELS dataset {0:} has resolution {1:} and offset {2:}".format(
            labels.shape, labels.attrs["resolution"], labels.attrs["offset"]
        )
    )
    cont = np.unique(labels)
    hist = np.histogram(labels, bins=list(cont) + [cont[-1] + 0.1])
    logging.info("LABELS contains ids {0:} in freq {1:}".format(cont, hist[0]))
    #   compute padding
    padding_before = (
        (
            (
                np.array(labels.attrs["offset"])
                - np.array(labels.attrs["resolution"]) / 2.0
            )
            + np.array(raw.attrs["resolution"] / 2.0)
        )
        / np.array(labels.attrs["resolution"])
    ).astype(np.int)
    padding_after = (
        2 * np.array(raw.shape) - padding_before - np.array(labels.shape)
    ).astype(np.int)
    padding = tuple((b, a) for b, a in zip(padding_before, padding_after))
    bg_label = 18446744073709551613
    logging.info(
        "padding LABELS with {0:} to match shape of upscaled RAW, padding value {1:} and relabeling "
        "using mapping {2:} to {3:}".format(
            padding, bg_label, range(len(mapping)), mapping
        )
    )
    # labels_padded = np.pad(labels, padding, 'constant', constant_values=bg_label)
    # numpy.pad has a bug when handling uint64, it is fixed in the current master so should be good with the next
    # numpy release (currently 1.14.3)
    labels_padded = (
        np.ones(tuple([rs * 2 for rs in raw.shape]), dtype=np.uint64) * bg_label
    )
    up_shape = labels_padded.shape
    labels_padded[
        padding[0][0] : -padding[0][1],
        padding[1][0] : -padding[1][1],
        padding[2][0] : -padding[2][1],
    ] = mapping[np.array(labels)]
    cont_relabeled = np.unique(labels_padded)
    hist_relabeled = np.histogram(
        labels_padded, bins=list(cont_relabeled) + [cont_relabeled[-1] + 0.1]
    )
    del cont_relabeled
    logging.info(
        "padded LABELS contains ids {0:} in freq {1:}".format(
            cont_relabeled, hist_relabeled[0]
        )
    )
    add_ds(
        target,
        "volumes/labels/all",
        labels_padded,
        labels.chunks,
        list(labels.attrs["resolution"]),
        [0.0, 0.0, 0.0],
        orig_ids=list(hist[1]),
        orig_counts=list(hist[0]),
        relabeled_ids=list(hist_relabeled[1]),
        relabeled_counts=list(hist_relabeled[0]),
        mapping=list(mapping),
    )
    # masks
    labels_padded = labels_padded[::2, ::2, ::2]
    mask = labels_padded
    del labels_padded
    mask = np.not_equal(mask, bg_label, out=mask).astype(np.uint64)

    add_ds(
        target,
        "volumes/masks/training",
        mask,
        labels.chunks,
        list(raw.attrs["resolution"]),
        [0.0, 0.0, 0.0],
    )

    for l in labelnames:
        if l not in specified_masks.keys():
            add_ds(
                target,
                "volumes/masks/" + l,
                mask,
                labels.chunks,
                list(raw.attrs["resolution"]),
                [0.0, 0.0, 0.0],
            )
        else:
            if specified_masks[l] == 0:
                label_mask = np.zeros(mask.shape, dtype=np.uint64)
            elif specified_masks[l] == 1:
                label_mask = mask
            elif isinstance(specified_masks[l], str) or isinstance(
                specified_masks[l], unicode
            ):
                assert orig[specified_masks[l]].shape == mask.shape
                label_mask = np.array(orig[specified_masks[l]]) * mask
            else:
                raise ValueError
            add_ds(
                target,
                "volumes/masks/" + l,
                label_mask,
                labels.chunks,
                list(raw.attrs["resolution"]),
                [0.0, 0.0, 0.0],
            )

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
            cont_label = np.unique(label_data)
            hist_label = np.histogram(
                label_data, bins=list(cont_label) + [cont_label[-1] + 0.1]
            )
            label_data_padded = np.ones(up_shape, dtype=np.uint64) * bg_label
            label_data_padded[
                padding[0][0] : -padding[0][1],
                padding[1][0] : -padding[1][1],
                padding[2][0] : -padding[2][1],
            ] = np.array(label_data)
            chunks = label_data.chunks
            res = label_data.attrs["resolution"]
            del label_data
            cont_label_relabeled = np.unique(label_data_padded)
            hist_label_relabeled = np.histogram(
                label_data_padded,
                bins=list(cont_label_relabeled) + [cont_label_relabeled][-1] + 0.1,
            )
            add_ds(
                target,
                "volumes/labels/" + l,
                label_data_padded,
                chunks,
                list(res),
                [0.0, 0.0, 0.0],
                orig_ids=list(hist_label[1]),
                orig_counts=list(hist_label[0]),
                relabeled_ids=list(hist_label_relabeled[1]),
                relabeled_counts=list(hist_label_relabeled[0]),
            )

    orig.close()


def main_multiscale_crop1(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop1_Periphery"
        "/Cell2_Crop1_1510x1510x1170+5961-280+83_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop1.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 4, 3, 10, 16, 2, 1, 1, 17, 11, 8, 30, 18, 19, 35, 9])
    # [0, mito lumen, mito membrane, MVB membrane, er membrane, plasma membrane, ECS, ECS, er lumen, MVB lumen,
    # vesicle membrane, microtubules, ERES membrane, ERES lumen, cytosol, vesicle lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop3(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop3_Mitos/Cell2_Crop3_1410x1410x1260"
        "+7295-305"
        "+3494_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop3.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 10, 2, 3, 8, 30, 17, 4, 11, 9, 18, 35, 1, 5, 19, 12, 13])
    # [0, er membrane, MVB membrane, plasma membrane, mito membrane, vesicle membrane, microtubules, er lumen,
    # mito lumen, MVB lumen, vesicle lumen, ERES membrane, cytosol, ECS, mito DNA, ERES lumen, lysosome membrane,
    # lysososme lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop4(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop4_Centrosome"
        "/Cell2_Crop4_1310x1310x1248+5595-305+2232_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop4.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array(
        [0, 20, 24, 10, 30, 28, 22, 23, 6, 35, 16, 7, 13, 11, 17, 21, 33, 32, 8, 9, 12]
    )
    # [0, NE membrane, HChrom , MVB membrane, microtubules, nucleoplasm, nuclear pore outside, nuclear pore inside,
    # golgi membrane, cytosol, er membrane, golgi lumen, lysosome lumen, MVB lumen, er lumen, NE lumen,
    # subidstal appendages, distal appendages, vesicle membrane, vesicle lumen, lysosome membrane]
    min_ad = 70.0
    max_ad = 204.0
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
            "EChrom": 0,
            "NEChrom": 1,
            "chromatin": 0,
            "microtubules": "volumes/masks/microtubules",
        },
    )


def main_multiscale_crop6(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop6_Ribosome"
        "/Cell2_Crop6_1260x1260x1260+2515-105+2494_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop6.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([30, 16, 17, 3, 4, 10, 11, 35, 18, 19, 30])
    # [0, er membrane, er lumen, mito membrane, mito lumen, MVB membrane, MVB lumen, cytosol, ERES membrane,
    # ERES lumen, microtubules]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig,
        target,
        labels,
        mapping,
        min_ad,
        max_ad,
        specified_masks={"ribosomes": 1},
        separate_datasets={"ribosomes": orig},
    )


def main_multiscale_crop7(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop7_PM"
        "/Cell2_Crop7_1310x1310x1170+7775-265-545_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop7.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 2, 2, 1, 35, 35, 30, 8, 9, 3, 4])
    # [0, plasma membrane, plasma membrane, ECS, cytosol, cytosol, microtubules, vesicle membrane, vesicle lumen,
    # mito membrane, mito lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig,
        target,
        labels,
        mapping,
        min_ad,
        max_ad,
        specified_masks={"ribosomes": 1},
        separate_datasets={"ribosomes": orig},
        src_label_name="volumes/labels/merged_ids",
    )


def main_multiscale_crop8(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop8_ERES001"
        "/Cell2_Crop8_1210x1210x1170+2880-225+714_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop8.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 18, 19, 30, 35, 17, 8, 9, 10])
    # [0, er membrane, ERES membrane, ERES lumen, microtubules, cytosol, er lumen, vesicle membrane, vesicle lumen,
    # MVB membrane]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop9(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop9_ERES002"
        "/Cell2_Crop9_1170x1170x1171+2365-115+1050_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop9.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 18, 3, 4, 35, 16, 17, 19, 10, 11, 30, 8, 9])
    # [0, ERES membrane, mito membrane, mito lumen, cytosol, er membrane, er lumen, ERES lumen, MVB membrane,
    # MVB lumen, microtubules, vesicle membrane, vesicle lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop13(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop13_ERES006"
        "/Cell2_Crop13_1170x1170x1170+4655+255+3402_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop13.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 18, 19, 10, 11, 35, 30, 20, 21, 36, 28, 17, 8, 9])
    # [0, er membrane, ERES membrane, ERES lumen, MVB  membrane, MVB lumen, cytosol, microtubules, nuclear envelope
    # membrane, nuclear envelope lumen, chromatin, nucleoplasm, er lumen, vesicle membrane, vesicle lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig,
        target,
        labels,
        mapping,
        min_ad,
        max_ad,
        specified_masks={
            "ribosomes": 1,
            "HChrom": 0,
            "NHChrom": 0,
            "EChrom": 0,
            "NEChrom": 0,
        },
        separate_datasets={"ribosomes": orig},
    )


def main_multiscale_crop14(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop14_ERES007"
        "/Cell2_Crop14_1170x1170x1171+5820-135+3863_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop14.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 19, 18, 30, 35, 17, 8, 9])
    # [0, er membrane, ERES lumen, ERES membrane, microtubules, cytosol, er lumen, vesicle membrane, vesicle lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop15(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop15_ERES008"
        "/Cell2_Crop15_1170x1170x1170+5620-65+3857_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop15.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 17, 18, 19, 35, 30])
    # [0, er membrane, er lumen, ERES membrane, ERES lumen, cytosol, microtubules]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop18(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop18_MVB"
        "/Cell2_Crop18_1210x1210x1170+1405-305+3200_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop18.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 30, 35, 16, 17, 10, 11, 8, 9])
    # [0, microtubules, cytosol, er membrane, er lumen, MVB membrane, MVB lumen, vesicle membrane, vesicle lumen]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop19(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop19_BadLD"
        "/Cell2_Crop19_1170x1170x1171+6390-110+4377_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop19.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 16, 17, 14, 15, 30, 8, 9, 35])
    # [0, er membrane, er lumen, LD membrane, LD lumen, microtubules, vesicle membrane, vesicle lumen, cytosol]
    min_ad = 70.0
    max_ad = 204.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop20(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/SUM159_Cell2_Crop20_LD001"
        "/Cell2_Crop20_1210x1210x1171+4200+240+4958_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop20.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 3, 4, 14, 15, 16, 17, 10, 11, 35])
    # [0, mito membrane, mito lumen, LD membrane, LD lumen, er membrane, er lumen, MVB membrane, MVB lumen, cytosol]
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop21(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/SUM159_Cell2_Crop21_LD002"
        "/Cell2_Crop21_1170x1170x1171+4055+125+5018_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop21.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 3, 4, 16, 17, 14, 15, 35])
    # [0, mito membrane, mito lumen, er membrane, er lumen, LD membrane LD lumen, cytosol]
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop22(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/SUM159_Cell2_Crop22_LD003"
        "/Cell2_Crop22_1180x1180x1170+3505-45+4845_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop22.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 35, 16, 17, 14, 15, 10, 11, 30])
    # [0, cytosol, er membrane, er lumen, LD membrane, LD lumen, MVB membrane, MVB lumen, microtubules]
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def main_multiscale_crop31(labels, offset, datestr="020719"):
    orig = h5py.File(
        "/nrs/saalfeld/heinrichl/cell/Annotations020719/SUM159_Cell2_Crop22_LD003"
        "/Cell2_Crop22_1180x1180x1170+3505-45+4845_8nm.h5",
        "r",
    )
    target = z5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cell/multires/v{0:}_{1:}_8nm/crop22.n5".format(
            datestr, offset
        ),
        use_zarr_format=False,
    )
    mapping = np.array([0, 35, 16, 17, 14, 15, 10, 11, 30])
    # [0, cytosol, er membrane, er lumen, LD membrane, LD lumen, MVB membrane, MVB lumen, microtubules]
    min_ad = 172.0
    max_ad = 233.0
    main_multiscale(
        orig, target, labels, mapping, min_ad, max_ad, specified_masks={"ribosomes": 0}
    )


def run_main_multiscale():
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
        "centrosome",
        "distal_app",
        "subdistal_app",
        "ribosomes",
    ]
    offset = "o505x505x505_m1170x1170x1170"
    main_multiscale_crop1(labels, offset)
    # main_multiscale_crop7(labels, offset)
    # main_multiscale_crop14(labels, offset)
    # main_multiscale_crop22(labels, offset)
    # main_multiscale_crop8(labels, offset)
    # main_multiscale_crop19(labels, offset)
    # main_multiscale_crop9(labels, offset)
    # main_multiscale_crop13(labels, offset)
    # main_multiscale_crop15(labels, offset)
    # main_multiscale_crop18(labels, offset)
    # main_multiscale_crop20(labels, offset)
    # main_multiscale_crop21(labels, offset)

    # main_multiscale_crop3(labels, offset)
    # main_multiscale_crop4(labels, offset)
    # main_multiscale_crop6(labels, offset)


def run_main():
    logging.basicConfig(level=logging.INFO)

    main_cell2_crop1()
    main_cell2_crop3()
    main_cell2_crop6()
    main_cell2_crop7()
    main_cell2_crop8()
    main_cell2_crop9()
    main_cell2_crop13()
    main_cell2_crop14()
    main_cell2_crop15()
    main_cell2_crop18()
    main_cell2_crop19()
    main_cell2_crop20()
    main_cell2_crop21()
    main_cell2_crop22()


if __name__ == "__main__":
    run_main_multiscale()
