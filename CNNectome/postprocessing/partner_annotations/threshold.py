import zarr
import numcodecs
import os
import numpy as np
import logging
from CNNectome.utils import config_loader


def threshold(filename_src, dataset_src, filename_tgt, dataset_tgt, thr):

    srcf = zarr.open(filename_src, mode="r")
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = zarr.open(filename_tgt, mode="a")
    tgtf.create_dataset(
        name=dataset_tgt,
        shape=srcf[dataset_src].shape,
        compressor=numcodecs.GZip(6),
        dtype="uint8",
        chunks=srcf[dataset_src].chunks,
    )
    ds = srcf[dataset_src][:]
    print(np.sum(ds > thr))
    print(np.min(ds))
    print(np.max(ds))
    tgtf[dataset_tgt][:] = (srcf[dataset_src][:] > thr).astype(np.uint8)
    if "offset" in srcf[dataset_src].attrs.keys():
        tgtf[dataset_tgt].attrs["offset"] = srcf[dataset_src].attrs["offset"]


def main():
    thrs_mult = [[127, 42]]
    samples = ["A", "B", "C", "A+", "B+", "C+"]
    setups_path = config_loader.get_config()["synapses"]["training_setups_path"]
    filename_src = os.path.join(setups_path,
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5"
    )
    dataset_srcs = [
        "predictions_it100000/cleft_dist_cropped",
        "predictions_it100000/cleft_dist_cropped",
    ]
    filename_tgt = os.path.join(setups_path,
        "pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5"
    )
    dataset_tgts = [
        "predictions_it100000/cleft_dist_cropped_thr{0:}",
        "predictions_it100000/cleft_dist_cropped_thr{0:}",
    ]

    for sample in samples:
        logging.info("thresholding sample {0:}".format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt, thr in zip(dataset_srcs, dataset_tgts, thrs):
                logging.info("    dataset {0:} at {1:}".format(ds_src, thr))
                threshold(
                    filename_src.format(sample),
                    ds_src,
                    filename_tgt.format(sample),
                    ds_tgt.format(thr),
                    thr,
                )


def run():
    filepath = os.path.join(config_loader.get_config()["synapses"]["training_setups_path"],
                            "cremi_all/cremi_all_0116_01/prediction_cremi_warped_sampleC+_200000.n5")
    dataset = "syncleft_dist"
    thr = 0.0
    dataset_tgt = "syncleft_dist_thr{0:}".format(thr)
    threshold(filepath, dataset, filepath, dataset_tgt, thr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
