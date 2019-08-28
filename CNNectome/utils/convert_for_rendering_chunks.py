import sys

sys.path.append("/groups/saalfeld/home/heinrichl/Projects/git_repos/gunpowder")
import z5py
import h5py
import numpy as np
import collections
from .label import Label

TRANSPARENT = 18446744073709551615


def add_container(
    target,
    name,
    shape,
    chunks=(128, 128, 128),
    dtype="uint64",
    resolution=(4, 4, 4),
    offset=(0, 0, 0),
):
    if name not in target:
        ds = target.create_dataset(
            name, shape=shape, chunks=chunks, dtype=dtype, compression="gzip"
        )
        target[name].attrs["resolution"] = resolution
        target[name].attrs["offset"] = offset


def add_data_combined_block(
    coord, blocksize, target, name, data, all_data, labelid, thr
):
    s = tuple(
        slice(c, min(c + b, maxs), None)
        for c, b, maxs in zip(coord, blocksize, target[name].shape)
    )
    data = np.array(data[s])
    data = (data > thr).astype(np.bool)
    all_data[data] = labelid
    data = data.astype(np.uint64)
    data[data == False] = TRANSPARENT
    target[name][s] = data.astype(np.uint64)
    return all_data


def add_data_asis_block(coord, blocksize, target, name, data):
    print("...{0:}".format(name))
    s = tuple(
        slice(c, min(c + b, maxs), None)
        for c, b, maxs in zip(coord, blocksize, target[name].shape)
    )

    if data.shape == target[name].shape:
        data = np.array(data[s])
    target[name][s] = data


def add_labels_blockwise(coord, blocksize, labels, target, orig):
    bs = [
        min(c + b, maxs) - c
        for c, b, maxs in zip(coord, blocksize, target["volumes/labels/all"].shape)
    ]

    block_data_all = np.ones(bs, dtype=np.uint64) * TRANSPARENT
    for label in labels:
        print("...{0:}".format(label.labelname))
        block_data_all = add_data_combined_block(
            coord,
            bs,
            target,
            "volumes/labels/{0:}".format(label.labelname),
            orig[label.labelname],
            block_data_all,
            label.targetid,
            thr=label.thr,
        )
    add_data_asis_block(coord, bs, target, "volumes/labels/all", block_data_all)


def main():

    orig = z5py.File(
        "/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_55000.n5",
        use_zarr_format=False,
    )

    # labels_combined_all = [('cell', 3, 128),
    #                        ('plasma_membrane', 2, 128),
    #                        ('er', 5, 128),
    #                        ('ERES', 7, 128),
    #                        ('mito', 9, 128),
    #                        ('MVB', 11, 128),
    #                        ('vesicles', 13, 128),
    #                        ('microtubules', 14, 128)]
    data_sources = [
        "hela_cell2_crop1_122018",
        "hela_cell2_crop3_122018",
        "hela_cell2_crop6_122018",
        "hela_cell2_crop7_122018",
        "hela_cell2_crop8_122018",
        "hela_cell2_crop9_122018",
        "hela_cell2_crop13_122018",
        "hela_cell2_crop14_122018",
        "hela_cell2_crop15_122018",
        "hela_cell2_crop18_122018",
        "hela_cell2_crop19_122018",
        "hela_cell2_crop20_122018",
        "hela_cell2_crop21_122018",
        "hela_cell2_crop22_122018",
    ]
    ribo_sources = [
        "hela_cell2_crop6_122018",
        "hela_cell2_crop7_122018",
        "hela_cell2_crop13_122018",
    ]
    labels = []
    labels.append(Label("ecs", 1, 1, thr=128, data_sources=data_sources))
    labels.append(Label("plasma_membrane", 2, 2, thr=128, data_sources=data_sources))
    labels.append(Label("mito", (3, 4, 5), 4, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "mito_membrane",
            3,
            3,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "mito_DNA",
            5,
            5,
            thr=128,
            scale_loss=False,
            scale_key=labels[-2].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("vesicle", (8, 9), 9, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "vesicle_membrane",
            8,
            8,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("MVB", (10, 11), 11, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "MVB_membrane",
            10,
            10,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("lysosome", (12, 13), 13, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "lysosome_membrane",
            12,
            12,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("LD", (14, 15), 15, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "LD_membrane",
            14,
            14,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label("er", (16, 17, 18, 19, 20, 21), 17, thr=133, data_sources=data_sources)
    )
    labels.append(
        Label(
            "er_membrane",
            (16, 18, 20),
            16,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(Label("ERES", (18, 19), 19, thr=133, data_sources=data_sources))
    labels.append(
        Label(
            "ERES_membrane",
            18,
            18,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label("nucleus", (20, 21, 24, 25), 25, thr=133, data_sources=data_sources)
    )
    labels.append(
        Label(
            "NE",
            (20, 21),
            21,
            thr=133,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "NE_membrane",
            20,
            20,
            thr=123,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "chromatin",
            24,
            24,
            thr=128,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )
    labels.append(
        Label(
            "nucleoplasm",
            25,
            25,
            thr=128,
            scale_loss=False,
            scale_key=labels[-1].scale_key,
            data_sources=data_sources,
        )
    )  # stop using this
    labels.append(Label("microtubules", 26, 26, thr=128, data_sources=data_sources))
    labels.append(Label("ribosomes", 1, 28, thr=128, data_sources=ribo_sources))

    # labels_combined_all = [('cell', 3, 128),
    #                       ('plasma_membrane', 2, 128),
    #                       ('er', 5, 133),
    #                       ('ERES', 7, 133),
    #                       ('mito', 9, 133),
    #                       ('MVB', 11, 133),
    #                       ('vesicles', 13, 133),
    #                       ('er_membrane', 4,  123),
    #                       ('ERES_membrane', 6, 123),
    #                       ('mito_membrane', 8, 123),
    #                       ('MVB_membrane', 10, 123),
    #                       ('vesicles_membrane', 13, 123),
    #                       ('microtubules', 14, 128)]
    shape = orig[labels[0].labelname].shape
    # shape = orig[labels_combined_all[0][0]].shape
    res = [4.0, 4.0, 4.0]
    offset = [0.0, 0.0, 0.0]
    target = z5py.File(
        "/nrs/saalfeld/heinrichl/cell/gt122018/setup01/run02/test2_55000_render_adapthr.n5",
        use_zarr_format=False,
    )
    orig_raw = z5py.File(
        "/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5",
        use_zarr_format=False,
    )
    if "volumes" not in list(target.keys()):
        target.create_group("volumes")
    if "labels" not in target["volumes"]:
        target["volumes"].create_group("labels")
    for label in labels:
        add_container(
            target,
            "volumes/labels/{0:}".format(label.labelname),
            shape,
            resolution=res,
            offset=offset,
        )
    add_container(target, "volumes/labels/all", shape, resolution=res, offset=offset)
    add_container(
        target, "volumes/raw", shape, dtype="uint8", resolution=res, offset=offset
    )
    blocksize = tuple(np.array([128, 128, 128]) * 5)
    for z in range(0, shape[0], blocksize[0]):
        for y in range(0, shape[1], blocksize[1]):
            for x in range(0, shape[2], blocksize[2]):
                print("Processing block at {0:}, {1:}, {2:}".format(z, y, x))
                add_labels_blockwise((z, y, x), blocksize, labels, target, orig)
                add_data_asis_block(
                    (z, y, x), blocksize, target, "volumes/raw", orig_raw["volumes/raw"]
                )


if __name__ == "__main__":
    main()
