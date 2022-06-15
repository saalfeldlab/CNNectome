import zarr
import numcodecs
import os
import numpy as np
import scipy.ndimage
import logging
import scipy.ndimage
from CNNectome.utils import config_loader


def cc2(
    filename_src, dataset_src_high_thr, dataset_src_low_thr, filename_tgt, dataset_tgt
):
    srcf = zarr.open(filename_src, mode="r")
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    assert (
        srcf[dataset_src_high_thr].attrs["offset"]
        == srcf[dataset_src_low_thr].attrs["offset"]
    )
    assert srcf[dataset_src_high_thr].shape == srcf[dataset_src_low_thr].shape
    tgtf = zarr.open(filename_tgt, mode="a")
    tgtf.empty(
        name=dataset_tgt,
        shape=srcf[dataset_src_high_thr].shape,
        compressor=numcodecs.GZip(6),
        dtype="uint64",
        chunks=srcf[dataset_src_high_thr].chunks,
    )
    data_high_thr = np.array(srcf[dataset_src_high_thr][:])
    data_low_thr = np.array(srcf[dataset_src_low_thr][:])
    tgt = np.ones(data_low_thr.shape, dtype=np.uint64)
    maxid = scipy.ndimage.label(data_low_thr, output=tgt)
    maxes = scipy.ndimage.maximum(
        data_high_thr, labels=tgt, index=list(range(1, maxid + 1))
    )
    maxes = np.array([0] + list(maxes))
    factors = maxes[tgt]

    tgt *= factors.astype(np.uint64)
    maxid = scipy.ndimage.label(tgt, output=tgt)
    tgtf[dataset_tgt][:] = tgt.astype(np.uint64)
    tgtf[dataset_tgt].attrs["offset"] = srcf[dataset_src_high_thr].attrs["offset"]
    tgtf[dataset_tgt].attrs["max_id"] = maxid


def main():
    thrs_mult = [[127, 42]]
    samples = ["A", "B", "C", "A+", "B+", "C+"]
    filename_src = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5",
    )
    dataset_srcs = ["predictions_it100000/cleft_dist_cropped_thr{0:}"]
    filename_tgt = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5",
    )
    dataset_tgts = ["predictions_it100000/cleft_dist_cropped_thr{0:}_cc{1:}"]

    for sample in samples:
        logging.info("finding connected components for sample {0:}".format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt in zip(dataset_srcs, dataset_tgts):
                logging.info("    dataset {0:}".format(ds_tgt.format(*thrs)))
                cc2(
                    filename_src.format(sample),
                    ds_src.format(thrs[0]),
                    ds_src.format(thrs[1]),
                    filename_tgt.format(sample),
                    ds_tgt.format(*thrs),
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
