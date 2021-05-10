from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import AddDistance, TanhSaturate, CombineDistances, IntensityCrop, Sum, CropArray
from gunpowder.ext import zarr
from gunpowder.compat import ensure_str
import CNNectome.utils.config_loader
import CNNectome.utils.label
import CNNectome.utils.cosem_db
import fuse
import tensorflow as tf
import math
import time
import json
import os
import numpy as np
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Sequence, Tuple


def get_label_ids_by_category(crop: Dict[str, Any],
                              category: str) -> List[int]:
    """
    Get all label ids from a crop that belong to a certain category

    Args:
        crop: Instance of an entry in the crop database.
        category: one of "present_annotated", "present_unannotated", "absent_annotated", "present_partial_annotation"

    Returns:
        All label ids that belong to the `category` for that crop.
    """
    return [ll[0] for ll in crop['labels'][category]]


def get_all_annotated_label_ids(crop: Dict[str, Any]) -> List[int]:
    """
    Get all label ids from a crop that are annotated (regardless of whether the labelid is present or absent)

    Args:
        crop: Instance of an entry in the crop database.

    Returns:
        All label ids that are annotated in the crop.
    """
    return get_label_ids_by_category(crop, "present_annotated") + get_label_ids_by_category(crop, "absent_annotated")


def get_crop_size(crop: Dict[str, Any]) -> int:
    """
    Compute the size of a crop in voxels (at annotation resolution).

    Args:
        crop: Instance of an entry in the crop database.

    Returns:
        Total number of voxels (at annotation resolution) in the given crop.
    """
    return np.prod(list(crop["dimensions"].values()))


def get_all_labelids(labels: List[CNNectome.utils.label.Label]) -> List[int]:
    """
    Generate list of all label ids occurring in the given list of labels.

    Args:
        labels: List of the labels from which to combine the label_ids.

    Returns:
        Combined list of all the label_ids from the labels in the list.
    """
    all_labelids = []
    for label in labels:
        all_labelids += list(label.labelid)
    return all_labelids


def prioritized_sampling_probabilities(crop_sizes: Union[Sequence[int], np.ndarray],
                                       indicator: Union[Sequence[bool], np.ndarray],
                                       prob_prioritized: float) -> List[float]:
    """
    Compute probabilities for sampling from each individual crop when using the prioritized sampling scheme. Crops
    that are prioritized will be sampled from with probability `prob_prioritized`. All other crops will be sampled
    from with probability 1 - `prob_prioritized`. Within each collection the sampling probability is proportional to
    the crop size.

    Args:
        crop_sizes: List of total size of each crop.
        indicator: List with value indicating for each crop whether it should be prioritized. Crops should follow same
                   ordering as in `crop_sizes`.
        prob_prioritized: Probability with which to sample from one of the prioritized crops.

    Returns:
        Probabilities for sampling from each crop, given in same order as `crop_sizes` and `indicator`.
    """
    crop_sizes_np = np.array(crop_sizes)
    indicator_np = np.array(indicator)
    prob_present = (
        prob_prioritized
        * crop_sizes_np
        * indicator_np
        / np.sum(crop_sizes_np[indicator_np])
    )
    prob_absent = (
        (1 - prob_prioritized)
        * crop_sizes_np
        * np.logical_not(indicator_np)
        / np.sum(crop_sizes_np[np.logical_not(indicator_np)])
    )
    return list(prob_present + prob_absent)


def is_prioritized(crop: Dict[str, Any],
                   prioritized_label: CNNectome.utils.label.Label) -> bool:
    """
    Determines whether a crop should be prioritized for training depending on whether it contains examples of the
    specified label.

    Args:
        crop: Instance of an entry in the crop database.
        prioritized_label: Other label that should be present in the crop for it to be considered prioritized.

    Returns:
        True if crop contains examples of specified `prioritized_label`, false otherwise.
    """
    present = set(get_label_ids_by_category(crop, "present_annotated"))
    annotated = set(get_all_annotated_label_ids(crop))
    # treating generic_label as separate case might not be necessary?
    if prioritized_label.generic_label is not None:
        specific_labels_prioritized = set(prioritized_label.labelid) - set(prioritized_label.generic_label)
        if not specific_labels_prioritized.isdisjoint(present) and specific_labels_prioritized.issubset(annotated):
            prioritized_crop = True
        elif not set(prioritized_label.generic_label).isdisjoint(present) and set(
            prioritized_label.generic_label
        ).issubset(annotated):
            prioritized_crop = True
        else:
            prioritized_crop = False
    else:
        prioritized_crop = not set(prioritized_label.labelid).isdisjoint(present) and set(
            prioritized_label.labelid
        ).issubset(annotated)
    return prioritized_crop


def _label_filter(cond_f: Callable[[CNNectome.utils.label.Label], bool],
                  labels: List[CNNectome.utils.label.Label]) -> List[CNNectome.utils.label.Label]:
    """
    Filter `labels` according to the given condition.

    Args:
        cond_f: Function that given a label evaluated to a boolean. Labels evaluating to False will be filtered out.
        labels: List of labels to filter.

    Returns:
        Copy of `labels` without the elements that evaluated to False with `cond_f`.
    """
    return [ll for ll in labels if cond_f(ll)]


def _get_label(name: str, labels: List[CNNectome.utils.label.Label]) -> Optional[CNNectome.utils.label.Label]:
    """
    Finds the (first) element of the `labels` list whose attribute labelname is `name`.

    Args:
        name: The label with this labelname should be extracted from `labels`.
        labels: List of labels in which to look for label with labelname `name`.

    Returns:
        If label with labelname `name` is found in the list `labels` return it, otherwise None.
    """
    filtered = _label_filter(lambda l: l.labelname == name, labels)
    if len(filtered) > 0:
        return filtered[0]
    else:
        return None


def _make_crop_source(crop: Dict[str, Any],
                      data_dir: Optional[str],
                      subsample_variant: Optional[
                         Union[int, str]],
                      gt_version: str,
                      labels: List[CNNectome.utils.label.Label],
                      ak_raw: ArrayKey,
                      ak_labels: ArrayKey,
                      ak_labels_downsampled: ArrayKey,
                      ak_mask: ArrayKey,
                      input_size: Coordinate,
                      output_size: Coordinate,
                      voxel_size_input: Coordinate,
                      voxel_size: Coordinate,
                      crop_width: Coordinate,
                      keep_thr: float) -> gunpowder.batch_provider_tree.BatchProviderTree:
    """
    Generate a batch provider for a specific crop, including label data, raw data, generating per label mask,
    rejection based on `min_masked_voxels` and contrast scaling.

    Args:
        crop: Instance of an entry in the crop database.
        data_dir: Path to directory where data is stored. If None, read from config file.
        subsample_variant: If using raw data that has been subsampled from its original resolution,
                           `subsample_variant` is the name of the dataset in the group "volumes/subsampled/raw"
                           containing the subsampled raw data. If None, use the raw data at original resolution
                           from "volumes/raw/s0".
        gt_version: Version of groundtruth annotations, e.g. "v0003"
        labels: List of labels that the network needs to be trained for.
        ak_raw: array key for raw data
        ak_labels: array key for label data
        ak_labels_downsampled: array key for downsampled label data
        ak_mask: array key for mask
        input_size: Size of input arrays of network.
        output_size: Size of output arrays of network.
        voxel_size_input: Voxel size of the input arrays.
        voxel_size: Voxel size of the output arrays.
        crop_width: additional padding width on top of `output_size`
        keep_thr: threshold for ratio of voxels that need to be annotated for a batch to not be rejected.

    Returns:
        Gunpowder  batch provider tree for grabbing batches from the specified crop.
    """
    if data_dir is None:
        data_dir = CNNectome.utils.config_loader.get_config()["organelles"]["data_path"]
    n5file = zarr.open(ensure_str(os.path.join(data_dir, crop["parent"])), mode='r')
    blueprint_label_ds = "volumes/groundtruth/{version:}/crop{cropno:}/labels/{{label:}}"
    blueprint_labelmask_ds = "volumes/groundtruth/{version:}/crop{cropno:}/masks/{{label:}}"
    blueprint_mask_ds = "volumes/masks/groundtruth/{version:}"
    if subsample_variant is None:
        raw_ds = "volumes/raw/s0"
    else:
        raw_ds = "volumes/subsampled/raw/{0:}".format(subsample_variant)
    label_ds = blueprint_label_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    labelmask_ds = blueprint_labelmask_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    mask_ds = blueprint_mask_ds.format(version=gt_version.lstrip("v"))

    # add sources for all groundtruth labels
    all_srcs = []
    # We should really only be adding this with the above if statement, but need it for now because we need to
    # construct masks from it as separate labelsets contain zeros
    logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
        cropno=crop["number"], file=os.path.join(data_dir, crop["parent"]), ds=label_ds.format(label="all"), ak=ak_labels))
    all_srcs.append(
        ZarrSource(os.path.join(data_dir, crop["parent"]),
                   {ak_labels: label_ds.format(label="all")}
                   )
        + Pad(ak_labels, Coordinate(output_size) + crop_width)
        + DownSample(ak_labels, (2, 2, 2), ak_labels_downsampled)
    )

    for label in _label_filter(lambda l: l.separate_labelset, labels):
        if all(l in get_label_ids_by_category(crop, "present_annotated") for l in label.labelid):
            ds = label_ds.format(label=label.labelname)
            assert ds in n5file, "separate dataset {ds:} not present in file {file:}".format(ds=ds,
                                                                                             file=n5file.store.path)
        else:
            ds = label_ds.format(label="all")
        logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
            cropno=crop["number"], file=os.path.join(data_dir, crop["parent"]), ds=ds, ak=label.gt_key))
        all_srcs.append(ZarrSource(os.path.join(data_dir, crop["parent"]), {label.gt_key: ds}) +
                        Pad(label.gt_key, Coordinate(output_size) + crop_width))

    # add mask source per label
    labelmask_srcs = []
    for label in labels:
        labelmask_ds = labelmask_ds.format(label=label.labelname)
        if labelmask_ds in n5file:  # specified mask available:
            logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
                cropno=crop["number"], file=os.path.join(data_dir, crop["parent"]), ds=labelmask_ds, ak=label.mask_key))
            labelmask_srcs.append(ZarrSource(os.path.join(data_dir, crop["parent"]),
                                             {label.mask_key: labelmask_ds}
                                             )
                                  + Pad(label.mask_key, Coordinate(output_size) + crop_width))
        else:
            if label.generic_label is not None:
                specific_labels = list(set(label.labelid) - set(label.generic_label))
                generic_condition = (all(l in get_all_annotated_label_ids(crop) for l in label.generic_label) or
                                     all(l in get_all_annotated_label_ids(crop) for l in specific_labels))
            else:
                generic_condition = False

            if all(l in get_all_annotated_label_ids(crop) for l in label.labelid) or generic_condition:
                f = lambda val: ((val > 0) * 1).astype(np.bool)
            else:
                f = lambda val: ((val > 0) * 0).astype(np.bool)
            # This does not work because there are crops that are very close to each other. This would lead to
            # issues with masking
            # logging.debug("Adding LambdaSource {f:} for crop {cropno:}, providing {ak}".format(
            #     cropno=crop["number"], f=f, ak=label.mask_key))
            # labelmask_srcs.append(
            #     LambdaSource(
            #         f,
            #         label.mask_key,
            #         {label.mask_key: ArraySpec(voxel_size=voxel_size, interpolatable=False)}
            #     )
            # )
            all_srcs[0] += LambdaFilter(f, ak_labels_downsampled, target_key=label.mask_key, target_spec=ArraySpec(
                dtype=np.bool, interpolatable=False))
    all_srcs.extend(labelmask_srcs)

    # add raw source
    logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
        cropno=crop["number"], file=os.path.join(data_dir, crop["parent"]), ds=raw_ds, ak=ak_raw))
    raw_src = (
        ZarrSource(
            os.path.join(data_dir, crop["parent"]),
            {ak_raw: raw_ds},
            array_specs={ak_raw: ArraySpec(voxel_size=voxel_size_input)})
        + Pad(ak_raw, Coordinate(input_size), 0)
    )
    all_srcs.append(raw_src)

    # add gt mask source
    logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
        cropno=crop["number"], file=os.path.join(data_dir, crop["parent"]), ds=mask_ds, ak=ak_mask))
    mask_src = (
        ZarrSource(
            os.path.join(data_dir, crop["parent"]),
            {ak_mask: mask_ds},
            array_specs={ak_mask: ArraySpec(interpolatable=False, voxel_size=voxel_size)}
        )
    )
    all_srcs.append(mask_src)

    # combine all sources and pick a random location
    crop_src = (
        tuple(all_srcs)
        + MergeProvider()
        + RandomLocation()
        + Reject(ak_mask, min_masked=keep_thr)
               )

    # contrast adjustment
    contr_adj = n5file["volumes/raw/s0"].attrs["contrastAdjustment"]
    scale = 255.0 / (float(contr_adj["max"]) - float(contr_adj["min"]))
    shift = - scale * float(contr_adj["min"])
    logging.debug("Adjusting contrast with scale {scale:} and shift {shift:}".format(scale=scale, shift=shift))
    crop_src += IntensityScaleShift(ak_raw,
                                    scale,
                                    shift
                                    )
    return crop_src


def _network_setup(max_iteration: int,
                   ak_raw: ArrayKey,
                   ak_mask: ArrayKey,
                   labels: List[CNNectome.utils.label.Label]) -> Tuple[
                   Dict[str, str], int, Dict[str, gunpowder.array.ArrayKey], Dict[str, gunpowder.array.ArrayKey]]:
    """
    Setup the tensorflow network with its inputs and outputs to be used in the train node.

    Args:
        max_iteration:  Total number of iterations that network should be trained for.
        ak_raw: array key for raw data
        ak_mask: array key for mask
        labels: List of labels that the network needs to be trained for.

    Returns:
        Tuple of net_io_names, start_iteration, inputs and outputs.
        net_io_names: Mapping easily usable names to names of tensors in the tensorflow graph, read from json file
                      that was generated when making the tensorflow graph
        start_iteration: Starting iteration of this training. 0 if training this setup for the first time,
                         otherwise read from checkpoint file.
        inputs: Mapping names of input tensors in `net_io_names` to keys of corresponding arrays in gunpowder
                batch.
        outputs: Mapping names of output tensors in `net_io_names` to keys of corresponding arrays in gunpowder
                 batch.
    """
    # load net_io_names.json
    with open("net_io_names.json", "r") as f:
        net_io_names = json.load(f)

    # find checkpoint from previous training, start a new one if not found
    if tf.train.latest_checkpoint("."):
        start_iteration = int(tf.train.latest_checkpoint(".").split("_")[-1])
        if start_iteration >= max_iteration:
            logging.info("Network has already been trained for {0:} iterations".format(start_iteration))
        else:
            logging.info("Resuming training from {0:}".format(start_iteration))
    else:
        start_iteration = 0
        logging.info("Starting fresh training")

    # define network inputs
    inputs = dict()
    inputs[net_io_names["raw"]] = ak_raw
    inputs[net_io_names["mask"]] = ak_mask
    for label in labels:
        inputs[net_io_names["mask_" + label.labelname]] = label.mask_key
        inputs[net_io_names["gt_" + label.labelname]] = label.gt_dist_key
        if label.scale_loss or label.scale_key is not None:
            inputs[net_io_names["w_" + label.labelname]] = label.scale_key

    # define network outputs
    outputs = dict()
    for label in labels:
        outputs[net_io_names[label.labelname]] = label.pred_dist_key
    return net_io_names, start_iteration, inputs, outputs


def train_until(
    max_iteration: int,
    gt_version: str,
    labels: List[CNNectome.utils.label.Label],
    net_name: str,
    input_shape: Union[np.ndarray, List[int]],
    output_shape: Union[np.ndarray, List[int]],
    loss_name: str,
    balance_global: bool = False,
    data_dir: Optional[str] = None,
    prioritized_label: Optional[CNNectome.utils.label.Label] = None,
    dataset: Optional[str] = None,
    prob_prioritized: float = 0.5,
    completion_min: int = 6,
    dt_scaling_factor: int = 50,
    cache_size: int = 5,
    num_workers: int = 10,
    min_masked_voxels: Union[float, int] = 17561.,
    voxel_size_labels: Coordinate = Coordinate((2, 2, 2)),
    voxel_size: Coordinate = Coordinate((4, 4, 4)),
    voxel_size_input: Coordinate = Coordinate((4, 4, 4))
):
    """
    Training a tensorflow network to learn signed distance transforms of specified labels (organelles) using gunpowder.
    Training data is read from crops whose metadata are organized in a database.

    Args:
        max_iteration: Total number of iterations that network should be trained for.
        gt_version: Version of groundtruth annotations, e.g. "v0003".
        labels: List of labels that the network needs to be trained for.
        net_name: Filename of tensorflow meta graph definition.
        input_shape: Input shape of network.
        output_shape: Output shape of network.
        loss_name: Name of loss used as stored in net io names json file.
        balance_global: If Ture, use globabl balancing, i.e. weigh loss for each label using its `frac_pos` and
                        `frac_neg` attributes.
        data_dir: Path to directory where data is stored. If None, read from config file.
        prioritized_label: Label to use for prioritizing sampling from crops that contain examples of it. If None
                           (default), sample from each crop equally.
        dataset: Only consider crops that come from the specified dataset. If None (default), use all othwerwise
                 eligible training data.
        prob_prioritized: If `prioritized_label` is not None, this is the probability with which to sample from the
                          crops containing the label. Default is .5, which implies sampling equally from crops
                          containing the labels and all others.
        completion_min: Minimal completion status for a crop from the database to be added to the training.
        dt_scaling_factor: Scaling factor to divide distance transform by before applying nonlinearity tanh.
        cache_size: Cache size for queue grabbing batches.
        num_workers: Number of workers grabbing batches.
        min_masked_voxels: Minimum number of voxels in a batch that need to be part of the groundtruth annotation.
        voxel_size_labels: Voxel size of the annotated labels.
        voxel_size: Voxel size of the desired output.
        voxel_size_input: Voxel size of the raw input data.
    """

    keep_thr = float(min_masked_voxels) / np.prod(output_shape)
    one_vx_thr = 1. / np.prod(output_shape)
    max_distance = 2.76 * dt_scaling_factor

    ak_raw = ArrayKey("RAW")
    ak_labels = ArrayKey("GT_LABELS")
    ak_labels_downsampled = ArrayKey("GT_LABELS_DOWNSAMPLED")
    ak_mask = ArrayKey("MASK")
    ak_labelmasks_comb = ArrayKey("LABELMASKS_COMBINED")
    input_size = Coordinate(input_shape) * voxel_size_input
    output_size = Coordinate(output_shape) * voxel_size
    crop_width = Coordinate((max_distance,) * len(voxel_size_labels))
    crop_width = crop_width//voxel_size
    if crop_width == 0:
        crop_width *= voxel_size
    else:
        crop_width = (crop_width+(1,)*len(crop_width)) * voxel_size
    # crop_width = crop_width  # (Coordinate((max_distance,) * len(voxel_size_labels))/2 )

    db = CNNectome.utils.cosem_db.MongoCosemDB(gt_version=gt_version)
    collection = db.access("crops", db.gt_version)
    db_filter = {"completion": {"$gte": completion_min}}
    if dataset is not None:
        db_filter['dataset_id'] = dataset
    skip = {"_id": 0, "number": 1, "labels": 1, "dataset_id": 1, "parent":1, "dimensions": 1}

    net_io_names, start_iteration, inputs, outputs = _network_setup()

    # construct batch request
    request = BatchRequest()
    request.add(ak_labels, output_size, voxel_size=voxel_size_labels)
    request.add(ak_labels_downsampled, output_size, voxel_size=voxel_size)
    request.add(ak_mask, output_size, voxel_size=voxel_size)
    request.add(ak_labelmasks_comb, output_size, voxel_size=voxel_size)
    request.add(ak_raw, input_size, voxel_size=voxel_size_input)
    for label in labels:
        if label.separate_labelset:
            request.add(label.gt_key, output_size, voxel_size=voxel_size_labels)
        request.add(label.gt_dist_key, output_size, voxel_size=voxel_size)
        request.add(label.pred_dist_key, output_size, voxel_size=voxel_size)
        request.add(label.mask_key, output_size, voxel_size=voxel_size)
        if label.scale_loss:
            request.add(label.scale_key, output_size, voxel_size=voxel_size)

    # specify specs for output
    array_specs_pred = dict()
    for label in labels:
        array_specs_pred[label.pred_dist_key] = ArraySpec(voxel_size=voxel_size,
                                                          interpolatable=True)
    # specify snapshot data layout
    snapshot_data = dict()
    snapshot_data[ak_raw] = "volumes/raw"
    snapshot_data[ak_mask] = "volumes/masks/all"
    if len(_label_filter(lambda l: not l.separate_labelset, labels)) > 0:
        snapshot_data[ak_labels] = "volumes/labels/gt_labels"
    for label in _label_filter(lambda l: l.separate_labelset, labels):
        snapshot_data[label.gt_key] = "volumes/labels/gt_"+label.labelname
    for label in labels:
        snapshot_data[label.gt_dist_key] = "volumes/labels/gt_dist_" + label.labelname
        snapshot_data[label.pred_dist_key] = "volumes/labels/pred_dist_" + label.labelname
        snapshot_data[label.mask_key] = "volumes/masks/" + label.labelname

    # specify snapshot request
    snapshot_request = BatchRequest()

    crop_srcs = []
    crop_sizes = []
    if prioritized_label is not None:
        crop_prioritized_label_indicator = []

    for crop in collection.find(db_filter, skip):
        if len(set(get_all_annotated_label_ids(crop)).intersection(set(get_all_labelids(labels)))) > 0:
            logging.info("Adding crop number {0:}".format(crop["number"]))
            if voxel_size_input != voxel_size:
                for subsample_variant in range(int(np.prod(voxel_size_input/voxel_size))):
                    crop_srcs.append(
                        _make_crop_source(crop, data_dir, subsample_variant, gt_version, labels, ak_raw, ak_labels,
                                          ak_labels_downsampled, ak_mask, input_size, output_size, voxel_size_input,
                                          voxel_size, crop_width, keep_thr))
                    crop_sizes.append(get_crop_size(crop))
                if prioritized_label is not None:
                    crop_prioritized = is_prioritized(crop, prioritized_label)
                    logging.info(f"Crop {crop['number']} is {'not ' if not crop_prioritized else ''}prioritized")
                    crop_prioritized_label_indicator.extend(
                        [crop_prioritized] * int(np.prod(voxel_size_input/voxel_size))
                    )
            else:
                crop_srcs.append(_make_crop_source(crop, data_dir, None, gt_version, labels, ak_raw, ak_labels,
                                                   ak_labels_downsampled, ak_mask, input_size, output_size,
                                                   voxel_size_input, voxel_size, crop_width, keep_thr))
                crop_sizes.append(get_crop_size(crop))
                if prioritized_label is not None:
                    crop_prioritized = is_prioritized(crop, prioritized_label)
                    logging.info(f"Crop {crop['number']} is {'not ' if not crop_prioritized else ''}prioritized")
                    crop_prioritized_label_indicator.append(crop_prioritized)

    if prioritized_label is not None:
        sampling_probs = prioritized_sampling_probabilities(
            crop_sizes, crop_prioritized_label_indicator, prob_prioritized
        )
    else:
        sampling_probs = crop_sizes
    print(sampling_probs)
    pipeline = (tuple(crop_srcs)
                + RandomProvider(sampling_probs)
                )

    pipeline += Normalize(ak_raw, 1.0/255)
    pipeline += IntensityCrop(ak_raw, 0., 1.)

    # augmentations
    pipeline = (pipeline
                + fuse.SimpleAugment()
                + fuse.ElasticAugment(voxel_size,
                                      (100, 100, 100),
                                      (10., 10., 10.),
                                      (0, math.pi / 2.),
                                      spatial_dims=3,
                                      subsample=8
                                      )
                + fuse.IntensityAugment(ak_raw, 0.25, 1.75, -0.5, 0.35)
                + GammaAugment(ak_raw, 0.5, 2.)
                )

    pipeline += IntensityScaleShift(ak_raw, 2, -1)

    # label generation
    for label in labels:
        pipeline += AddDistance(
            label_array_key=label.gt_key,
            distance_array_key=label.gt_dist_key,
            mask_array_key=label.mask_key,
            add_constant=label.add_constant,
            label_id=label.labelid,
            factor=2,
            max_distance=max_distance,
        )

    # combine distances for centrosomes

    centrosome = _get_label("centrosome", labels)
    microtubules = _get_label("microtubules", labels)
    microtubules_out = _get_label("microtubules_out", labels)
    subdistal_app = _get_label("subdistal_app", labels)
    distal_app = _get_label("distal_app", labels)

    # add the centrosomes to the microtubules
    if microtubules_out is not None and centrosome is not None:
        pipeline += CombineDistances(
            (microtubules_out.gt_dist_key, centrosome.gt_dist_key),
            microtubules_out.gt_dist_key,
            (microtubules_out.mask_key, centrosome.mask_key),
            microtubules_out.mask_key
        )
    if microtubules is not None and centrosome is not None:
        pipeline += CombineDistances(
            (microtubules.gt_dist_key, centrosome.gt_dist_key),
            microtubules.gt_dist_key,
            (microtubules.mask_key, centrosome.mask_key),
            microtubules.mask_key
        )

    # add the distal_app and subdistal_app to the centrosomes
    if centrosome is not None and distal_app is not None and subdistal_app is not None:
        pipeline += CombineDistances(
            (distal_app.gt_dist_key, subdistal_app.gt_dist_key, centrosome.gt_dist_key),
            centrosome.gt_dist_key,
            (distal_app.mask_key, subdistal_app.mask_key, centrosome.mask_key),
            centrosome.mask_key
        )

    arrays_that_need_to_be_cropped = []

    for label in labels:
        arrays_that_need_to_be_cropped.append(label.gt_key)
        arrays_that_need_to_be_cropped.append(label.gt_dist_key)
        arrays_that_need_to_be_cropped.append(label.mask_key)
    arrays_that_need_to_be_cropped.append(ak_labels)
    arrays_that_need_to_be_cropped.append(ak_labels_downsampled)
    arrays_that_need_to_be_cropped = list(set(arrays_that_need_to_be_cropped))
    for ak in arrays_that_need_to_be_cropped:
        pipeline += CropArray(ak, crop_width, crop_width)

    for label in labels:
        pipeline += TanhSaturate(label.gt_dist_key, dt_scaling_factor)

    for label in _label_filter(lambda l: l.scale_loss, labels):
        if balance_global:
            pipeline += BalanceGlobalByThreshold(
                label.gt_dist_key,
                label.scale_key,
                label.frac_pos,
                label.frac_neg
            )
        else:
            pipeline += BalanceByThreshold(
                label.gt_dist_key,
                label.scale_key,
                mask=(label.mask_key, ak_mask)
                )
    pipeline += Sum([l.mask_key for l in labels], ak_labelmasks_comb, sum_array_spec=ArraySpec(
                    dtype=np.uint8, interpolatable=False))
    pipeline += Reject(ak_labelmasks_comb, min_masked=one_vx_thr)

    pipeline = (pipeline
                + PreCache(cache_size=cache_size,
                           num_workers=num_workers)
                + Train(net_name,
                        optimizer=net_io_names["optimizer"],
                        loss=net_io_names[loss_name],
                        inputs=inputs,
                        summary=net_io_names["summary"],
                        log_dir="log",
                        outputs=outputs,
                        gradients={},
                        log_every=10,
                        save_every=500,
                        array_specs=array_specs_pred,
                        )
                + Snapshot(snapshot_data,
                           every=500,
                           output_filename="batch_{iteration}.hdf",
                           output_dir="snapshots/",
                           additional_request=snapshot_request,
                           )
                + PrintProfilingStats(every=50)
                )

    logging.info("Starting training...")
    with build(pipeline) as pp:
        for i in range(start_iteration, max_iteration+1):
            start_it = time.time()
            pp.request_batch(request)
            time_it = time.time() - start_it
            logging.info("it{0:}: {1:}".format(i+1, time_it))
    logging.info("Training finished")
