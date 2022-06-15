import zarr
import numcodecs
import os
import numpy as np
import scipy.ndimage
import logging
from CNNectome.utils import config_loader


def cc(filename_src, dataset_src, filename_tgt, dataset_tgt):
    srcf = zarr.open(filename_src, mode="r")
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = zarr.open(filename_tgt, mode="a")
    tgtf.empty(
        name=dataset_tgt,
        shape=srcf[dataset_src].shape,
        compressor=numcodecs.GZip(6),
        dtype="uint64",
        chunks=srcf[dataset_src].chunks,
    )
    data = np.array(srcf[dataset_src][:])
    tgt = np.ones(data.shape, dtype=np.uint64)
    maxid = scipy.ndimage.label(data, output=tgt)
    tgtf[dataset_tgt][:] = tgt.astype(np.uint64)
    if "offset" in srcf[dataset_src].attrs.keys():
        tgtf[dataset_tgt].attrs["offset"] = srcf[dataset_src].attrs["offset"]
    tgtf[dataset_tgt].attrs["max_id"] = maxid


def main():
    thrs_mult = [[153, 76, 76], [127, 63, 63]]
    samples = ["B"]  # ['A+', 'B+', 'C+', 'A', 'B', 'C']
    filename_src = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5",
    )
    dataset_srcs = ["predictions_it400000/cleft_dist_cropped_thr{0:}"]
    filename_tgt = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5",
    )
    dataset_tgts = ["predictions_it400000/cleft_dist_cropped_thr{0:}_cc"]

    for sample in samples:
        logging.info("finding connected components for sample {0:}".format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt, thr in zip(dataset_srcs, dataset_tgts, thrs):
                logging.info("    dataset {0:}".format(ds_src.format(thr)))
                cc(
                    filename_src.format(sample),
                    ds_src.format(thr),
                    filename_tgt.format(sample),
                    ds_tgt.format(thr),
                )


def run():
    filepath = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "cremi_all/cremi_all_0116_01/prediction_cremi_warped_sampleC+_200000.n5",
    )
    dataset = "syncleft_dist_thr0.0"
    dataset_tgt = dataset + "_cc"
    cc(filepath, dataset, filepath, dataset_tgt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # run()
    filepath = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "cremi_all/cremi_all_0116_01/prediction_cremi_warped_sampleB_200000.n5",
    )
    dataset = "syncleft_dist_thr0.0"
    dataset_tgt = dataset + "_cc"
    filepath_tgt = "test.n5"
    cc(filepath, dataset, filepath_tgt, dataset_tgt)
