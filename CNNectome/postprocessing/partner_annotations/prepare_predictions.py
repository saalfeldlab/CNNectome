from . import connected_components_double_threshold
from . import crop
from . import threshold
import logging
import sys
import os
from CNNectome.utils import config_loader

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


def crop_main(it):
    setups_path = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"], "pre_and_post"
    )
    samples = ["A+", "B+", "C+", "A", "B", "C"]
    filename_src = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_srcs = [
        "predictions_it{0:}/cleft_dist".format(it),
        "predictions_it{0:}/pre_dist".format(it),
        "predictions_it{0:}/post_dist".format(it),
    ]
    filename_tgt = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_tgts = [
        "predictions_it{0:}/cleft_dist_cropped".format(it),
        "predictions_it{0:}/pre_dist_cropped".format(it),
        "predictions_it{0:}/post_dist_cropped".format(it),
    ]
    for sample in samples:
        logging.info("cropping sample {0:}".format(sample))
        off = offsets[sample]
        sh = shapes[sample]
        for ds_src, ds_tgt in zip(dataset_srcs, dataset_tgts):
            logging.info("   dataset {0:}".format(ds_src))
            crop.crop_to_seg(
                filename_src.format(sample),
                ds_src,
                filename_tgt.format(sample),
                ds_tgt,
                off,
                sh,
            )


def thr_main(it):
    thrs_mult = [[127, 42]]
    samples = ["A", "B", "C", "A+", "B+", "C+"]
    setups_path = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"], "pre_and_post"
    )
    filename_src = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_srcs = [
        "predictions_it{0:}/cleft_dist_cropped".format(it),
        "predictions_it{0:}/cleft_dist_cropped".format(it),
    ]
    filename_tgt = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_tgts = [
        "predictions_it{0:}".format(it) + "/cleft_dist_cropped_thr{0:}",
        "predictions_it{0:}".format(it) + "/cleft_dist_cropped_thr{0:}",
    ]

    for sample in samples:
        logging.info("thresholding sample {0:}".format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt, thr in zip(dataset_srcs, dataset_tgts, thrs):
                logging.info("    dataset {0:} at {1:}".format(ds_src, thr))
                threshold.threshold(
                    filename_src.format(sample),
                    ds_src,
                    filename_tgt.format(sample),
                    ds_tgt.format(thr),
                    thr,
                )


def cc_main(it):

    thrs_mult = [[127, 42]]
    samples = ["A", "B", "C", "A+", "B+", "C+"]
    setups_path = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"], "pre_and_post"
    )
    filename_src = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_srcs = ["predictions_it{0:}".format(it) + "/cleft_dist_cropped_thr{0:}"]
    filename_tgt = os.path.join(setups_path, "pre_and_post-v6.3/cremi/{0:}.n5")
    dataset_tgts = [
        "predictions_it{0:}".format(it) + "/cleft_dist_cropped_thr{0:}_cc{1:}"
    ]

    for sample in samples:
        logging.info("finding connected components for sample {0:}".format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt in zip(dataset_srcs, dataset_tgts):
                logging.info("    dataset {0:}".format(ds_tgt.format(*thrs)))
                connected_components_double_threshold.cc2(
                    filename_src.format(sample),
                    ds_src.format(thrs[0]),
                    ds_src.format(thrs[1]),
                    filename_tgt.format(sample),
                    ds_tgt.format(*thrs),
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    i = sys.argv[1]
    crop_main(i)
    thr_main(i)
    cc_main(i)
