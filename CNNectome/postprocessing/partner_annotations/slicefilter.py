import zarr
import numcodecs
import os
import numpy as np
import scipy.ndimage
from CNNectome.utils import config_loader

BG_VAL1 = 0xFFFFFFFFFFFFFFFD
BG_VAL2 = 0


def slicefilter(
    filename_src, dataset_src, filename_tgt, dataset_tgt, thr, dat_file=None
):

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
    tgt = np.array(srcf[dataset_src][:])

    ids, relabeled = np.unique(tgt, return_inverse=True)
    relabeled = relabeled.reshape(tgt.shape) + 1
    if BG_VAL1 in ids:
        relabeled[tgt == BG_VAL1] = 0
    if BG_VAL2 in ids:
        relabeled[tgt == BG_VAL2] = 0

    obj_slices = scipy.ndimage.measurements.find_objects(relabeled)
    set_to_bg = []
    for k, obs in enumerate(obj_slices):
        if not None:
            if relabeled[obs].shape[0] <= thr:
                set_to_bg.append(k + 1)

    tgt[np.isin(relabeled, set_to_bg)] = 0
    tgtf[dataset_tgt][:] = tgt.astype(np.uint64)
    tgtf[dataset_tgt].attrs["offset"] = srcf[dataset_src].attrs["offset"]


def main():
    thrs = [127, 63, 63]
    samples = ["A+", "B+", "C+"]
    # samples = ['B+', 'C+']
    slf = 1
    offsets = {
        "A+": (37 * 40, 1176 * 4, 955 * 4),
        "B+": (37 * 40, 1076 * 4, 1284 * 4),
        "C+": (37 * 40, 1002 * 4, 1165 * 4),
        "A": (38 * 40, 942 * 4, 951 * 4),
        "B": (37 * 40, 1165 * 4, 1446 * 4),
        "C": (37 * 40, 1032 * 4, 1045 * 4),
    }

    # segf_name = {'A+': 'sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81',
    #             'B+': 'sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87',
    #             'C+': 'sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75',
    #            }

    # filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/{0:}.n5'
    # dataset_src = 'main'
    # filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/{0:}_sizefiltered750.n5'
    # dataset_tgt = 'main'
    # dat_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_sizefilter750.dat'

    # filename_src = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample{0}.n5'
    # dataset_src = 'segmentations/multicut'
    # dataset_src = 'segmentations/mc_glia_global2'

    # filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{0:}_padded_20170424.aligned.0bg.n5'
    # dataset_src = 'volumes/labels/neuron_ids'

    # filename_src = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_new/sample{0:}.n5'
    # dataset_src = 'segmentation/multicut'
    # filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0}.n5'
    filename = os.path.join(config_loader.get_config()["synapses"]["cremieval_path"],"data2016-aligned/{0:}.n5")
    dataset_src = "volumes/labels/neuron_ids_constis_cropped"
    dataset_tgt = "volumes/labels/neuron_ids_constis_slf{0:}_cropped"
    # dat_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_constis_slicefilter{' \
    #           '1:}.dat'
    for sample in samples:
        print(sample)
        slicefilter(
            filename.format(sample),
            dataset_src,
            filename.format(sample),
            dataset_tgt.format(slf),
            slf,
        )  # , dat_file.format(sample, slf))


if __name__ == "__main__":
    main()
