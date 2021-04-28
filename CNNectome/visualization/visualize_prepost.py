import logging
import numpy as np
import h5py
import os
import warnings
from scipy.ndimage.morphology import distance_transform_edt
import csv
from CNNectome.utils import config_loader

logger = logging.getLogger(__name__)


def make_cleft_to_prepostsyn_neuron_id_dict(csv_files):
    cleft_to_pre = dict()
    cleft_to_post = dict()
    for csv_f in csv_files:
        f = open(csv_f, "r")
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[10]) != -1:
                try:
                    cleft_to_pre[int(row[10])].add(int(row[0]))
                except KeyError:
                    cleft_to_pre[int(row[10])] = {int(row[0])}
                try:
                    cleft_to_post[int(row[10])].add(int(row[5]))
                except KeyError:
                    cleft_to_post[int(row[10])] = {int(row[5])}
    return cleft_to_pre, cleft_to_post


def find_boundaries(labels):
    # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
    # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
    # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
    # bound.: 00000001000100000001000      2n - 1

    logger.debug("computing boundaries for %s", labels.shape)
    dims = len(labels.shape)
    in_shape = labels.shape
    out_shape = tuple(2 * s - 1 for s in in_shape)
    out_slices = tuple(slice(0, s) for s in out_shape)
    boundaries = np.zeros(out_shape, dtype=np.bool)
    logger.info("boundaries shape is %s", boundaries.shape)
    for d in range(dims):
        logger.info("processing dimension %d", d)
        shift_p = [slice(None)] * dims
        shift_p[d] = slice(1, in_shape[d])
        shift_n = [slice(None)] * dims
        shift_n[d] = slice(0, in_shape[d] - 1)
        diff = (labels[shift_p] - labels[shift_n]) != 0
        logger.info("diff shape is %s", diff.shape)
        target = [slice(None, None, 2)] * dims
        target[d] = slice(1, out_shape[d], 2)
        logger.info("target slices are %s", target)
        boundaries[target] = diff
    return boundaries


def normalize(distances, normalize_mode, normalize_args):
    if normalize_mode == "tanh":
        scale = normalize_args
        return np.tanh(distances / scale)
    else:
        raise NotImplementedError


def create_prepost_dt(
    clefts,
    labels,
    voxel_size,
    cleft_to_presyn_neuron_id,
    cleft_to_postsyn_neuron_id,
    bg_value=0,
    include_cleft=True,
    normalize_mode="tanh",
    normalize_args=50,
):
    max_distance = min(dim * vs for dim, vs in zip(clefts.shape, voxel_size))
    presyn_distances = -np.ones(clefts.shape, dtype=np.float) * max_distance
    postsyn_distances = -np.ones(clefts.shape, dtype=np.float) * max_distance

    contained_cleft_ids = np.unique(clefts)
    for cleft_id in contained_cleft_ids:
        if cleft_id != bg_value:
            d = -distance_transform_edt(clefts != cleft_id, sampling=voxel_size)
            try:
                pre_neuron_id = np.array(list(cleft_to_presyn_neuron_id[cleft_id]))
                pre_mask = np.any(
                    labels[..., None] == pre_neuron_id[None, ...], axis=-1
                )
                if include_cleft:
                    pre_mask = np.any([pre_mask, clefts == cleft_id], axis=0)
                presyn_distances[pre_mask] = np.max((presyn_distances, d), axis=0)[
                    pre_mask
                ]
            except KeyError:
                warnings.warn("No Key in Pre Dict %s" % str(cleft_id))
            try:
                post_neuron_id = np.array(list(cleft_to_postsyn_neuron_id[cleft_id]))
                post_mask = np.any(
                    labels[..., None] == post_neuron_id[None, ...], axis=-1
                )
                if include_cleft:
                    post_mask = np.any([post_mask, clefts == cleft_id], axis=0)
                postsyn_distances[post_mask] = np.max((postsyn_distances, d), axis=0)[
                    post_mask
                ]
            except KeyError:
                warnings.warn("No Key in Post Dict %s" % str(cleft_id))
    if normalize is not None:
        presyn_distances = normalize(
            presyn_distances, normalize_mode, normalize_args=normalize_args
        )
        postsyn_distances = normalize(
            postsyn_distances, normalize_mode, normalize_args=normalize_args
        )
        return presyn_distances, postsyn_distances


def create_dt(
    labels, target_file, voxel_size=(1, 1, 1), normalize_mode=None, normalize_args=None
):
    boundaries = 1.0 - find_boundaries(labels)
    print(np.sum(boundaries == 0))
    if np.sum(boundaries == 0) == 0:
        max_distance = min(dim * vs for dim, vs in zip(labels.shape, voxel_size))
        if np.sum(labels) == 0:
            distances = -np.ones(labels.shape, dtype=np.float32) * max_distance
        else:
            distances = np.ones(labels.shape, dtype=np.float32) * max_distance

    else:

        # get distances (voxel_size/2 because image is doubled)
        print("compute dt")
        distances = distance_transform_edt(
            boundaries, sampling=tuple(float(v) / 2 for v in voxel_size)
        )
        print("type conversion")
        distances = distances.astype(np.float32)

        # restore original shape
        print("downsampling")
        downsample = (slice(None, None, 2),) * len(voxel_size)
        distances = distances[downsample]

        print("signed dt")
        # todo: inverted distance
        distances[labels == 0] = -distances[labels == 0]

    distances = np.expand_dims(distances, 0)

    if normalize_mode is not None:
        print("normalizing")
        distances = normalize(distances, normalize_mode, normalize_args)
    print("saving")
    target_file.create_dataset("data", data=distances.squeeze())
    target_file.close()


def main():
    data_sources = ["A", "B", "C"]

    cremi_dir = config_loader.get_config()["synapses"]["cremi17_data_path"]
    csv_files = [
        os.path.join(cremi_dir, "cleft-partners_" + sample + "_2017.csv")
        for sample in data_sources
    ]
    hf = h5py.File(
        os.path.join(cremi_dir, "sample_B_padded_20170424.aligned.0bg.hdf"), "r"
    )
    clefts = np.array(hf["volumes/labels/clefts"][50:150, 1400:2400, 1900:2900])
    labels = np.array(hf["volumes/labels/neuron_ids"][50:150, 1400:2400, 1900:2900])
    raw = np.array(hf["volumes/raw"][50:150, 1400:2400, 1900:2900])
    pre_dict, post_dict = make_cleft_to_prepostsyn_neuron_id_dict(csv_files)
    print("data loaded")
    pre_dist, post_dist = create_prepost_dt(
        clefts, labels, (40, 4, 4), pre_dict, post_dict
    )
    print("dist computed")
    hf.close()
    tar = h5py.File("righthere.hdf", "w")
    tar.create_dataset("pre_dist", data=pre_dist)
    tar.create_dataset("post_dist", data=post_dist)
    tar.create_dataset("raw", data=raw)
    tar.close()


if __name__ == "__main__":
    main()
    # label_data = h5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_C_cleftsonly_bin.hdf',
    #                       'r')['volumes/labels/clefts']
    # print(label_data.shape)
    # target_file = h5py.File('/groups/saalfeld/saalfeldlab/larissa/data/gunpowder/cremi/gt_xz.h5', 'w')
    # if 'resolution' in label_data.attrs:
    #    voxel_size = tuple(label_data.attrs['resolution'])
    #    print('yes')
    # else:
    #    voxel_size = (4,4,40)

    # normalize_mode = 'tanh'
    # scale=50
    # print(voxel_size)
    ##print(np.sum(np.array(label_data)[:,1500,]))
    # create_dt(np.array(label_data)[:,1550-550:1550+550+1,:], target_file, voxel_size,
    #          normalize_mode=normalize_mode,
    #          normalize_args=scale)
