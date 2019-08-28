import z5py
import os
import logging

offsets_minicrop = {
    "A+": (37, 1676, 1598),
    "B+": (37, 2201, 3294),
    "C+": (37, 1702, 2135),
}
shapes = {
    "A+": (125, 1529, 1448),
    "B+": (125, 1701, 2794),
    "C+": (125, 1424, 1470),
    "A": (125, 1438, 1322),
    "B": (125, 1451, 2112),
    "C": (125, 1578, 1461),
}
offsets = {
    "A+": (37, 1176, 955),
    "B+": (37, 1076, 1284),
    "C+": (37, 1002, 1165),
    "A": (38, 942, 951),
    "B": (37, 1165, 1446),
    "C": (37, 1032, 1045),
}


def set_mask_to_zero(
    filename_src,
    dataset_src,
    filename_mask,
    dataset_mask,
    filename_tgt,
    dataset_tgt,
    offset,
    shape,
):
    logging.info("setting mask to zero for " + filename_src + "/" + dataset_src)
    srcf = z5py.File(filename_src, use_zarr_format=False)
    maskf = z5py.File(filename_mask, use_zarr_format=False)

    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = z5py.File(filename_tgt, use_zarr_format=False)
    grps = ""
    for grp in dataset_tgt.split("/")[:-1]:
        grps += grp
        if not os.path.exists(os.path.join(filename_tgt, grps)):
            tgtf.create_group(grps)
        grps += "/"
    chunk_size = tuple(min(c, s) for c, s in zip(srcf[dataset_src].chunks, shape))

    tgtf.create_dataset(
        dataset_tgt,
        shape=shape,
        compression="gzip",
        dtype=srcf[dataset_src].dtype,
        chunks=chunk_size,
    )
    a = srcf[dataset_src][:]
    a[maskf[dataset_mask][:] == 0] = 0
    tgtf[dataset_tgt][:] = a
    tgtf[dataset_tgt].attrs["offset"] = offset[::-1]


def main_seg():
    samples = ["A", "B", "C"]  # ['A', 'C', 'B+', 'C+']
    filename_src = "/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5"
    dataset_src = "volumes/labels/neuron_ids_constis_slf1_sf750_cropped"
    filename_mask = "/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5"
    dataset_mask = "volumes/labels/neuron_ids_gt_cropped"
    filename_tgt = "/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5"
    dataset_tgt = "volumes/labels/neuron_ids_constis_slf1_sf750_cropped_masked"
    for sample in samples:
        print(sample)
        off = offsets[sample]
        sh = shapes[sample]
        set_mask_to_zero(
            filename_src.format(sample),
            dataset_src,
            filename_mask.format(sample),
            dataset_mask,
            filename_tgt.format(sample),
            dataset_tgt,
            off,
            sh,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_seg()
