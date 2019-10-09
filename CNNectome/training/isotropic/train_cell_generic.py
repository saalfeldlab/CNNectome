from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import AddDistance, TanhSaturate, CombineDistances, IntensityCrop, Sum
from gunpowder.ext import zarr
from gunpowder.compat import ensure_str

import fuse
import tensorflow as tf
import math
import time
import json
import numpy as np
import logging
import pymongo


def get_label_ids_by_category(crop, category):
    return [l[0] for l in crop['labels'][category]]


def get_all_annotated_label_ids(crop):
    return get_label_ids_by_category(crop, "present_annotated") + get_label_ids_by_category(crop, "absent_annotated")


def get_crop_size(crop):
    return np.prod(list(crop["dimensions"].values()))


def get_all_labelids(labels):
    all_labelids = []
    for label in labels:
        all_labelids += list(label.labelid)
    return all_labelids


def train_until(
    max_iteration,
    gt_version,
    labels,
    net_name,
    input_shape,
    output_shape,
    loss_name,
    db_username,
    db_password,
    db_name="crops",
    completion_min=6,
    dt_scaling_factor=50,
    cache_size=5,
    num_workers=10,
    min_masked_voxels=17561.,
    voxel_size_labels=Coordinate((2, 2, 2)),
    voxel_size=Coordinate((4, 4, 4)),
    voxel_size_input=Coordinate((4, 4, 4))
):
    def label_filter(cond_f):
        return [ll for ll in labels if cond_f(ll)]

    def get_label(name):
        filter = label_filter(lambda l: l.labelname == name)
        if len(filter) > 0:
            return filter[0]
        else:
            return None

    def make_crop_source(crop, subsample_variant=None):
        n5file = zarr.open(ensure_str(crop["parent"]), mode='r')
        blueprint_label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/{{label:}}"
        blueprint_labelmask_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/masks/{{label:}}"
        blueprint_mask_ds = "volumes/masks/groundtruth/{version:}"
        if subsample_variant is None:
            raw_ds = "volumes/raw"
        else:
            raw_ds = "volumes/subsampled/raw/{0:}".format(subsample_variant)
        label_ds = blueprint_label_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
        labelmask_ds = blueprint_labelmask_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
        mask_ds = blueprint_mask_ds.format(version=gt_version.lstrip("v"))

        # add sources for all groundtruth labels
        all_srcs = []
        #if len(label_filter(lambda l: not l.separate_labelset)) > 0:
        # We should really only be adding this with the above if statement, but need it for now because we need to
        # construct masks from it as separate labelsets contain zeros
        logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
            cropno=crop["number"], file=crop["parent"], ds=label_ds.format(label="all"), ak=ak_labels))
        all_srcs.append(
            ZarrSource(crop["parent"],
                       {ak_labels: label_ds.format(label="all")}
                       )
            + Pad(ak_labels, Coordinate(output_size))
            + DownSample(ak_labels, (2, 2, 2), ak_labels_downsampled)
        )

        for label in label_filter(lambda l: l.separate_labelset):
            if all(l in get_label_ids_by_category(crop, "present_annotated") for l in label.labelid):
                ds = label_ds.format(label=label.labelname)
                assert ds in n5file, "separate dataset {ds:} not present in file {file:}".format(ds=ds,
                                                                                                 file=n5file.store.path)
            else:
                ds = label_ds.format(label="all")
            logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
                cropno=crop["number"], file=crop["parent"], ds=ds, ak=label.gt_key))
            all_srcs.append(ZarrSource(crop["parent"],
                                         {label.gt_key: ds}
                                         )
                              + Pad(label.gt_key, Coordinate(output_size)))

        # add mask source per label
        labelmask_srcs = []
        for label in labels:
            labelmask_ds = labelmask_ds.format(label=label.labelname)
            if labelmask_ds in n5file:  # specified mask available:
                logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
                    cropno=crop["number"], file=crop["parent"], ds=labelmask_ds, ak=label.mask_key))
                labelmask_srcs.append(ZarrSource(crop["parent"],
                                                 {label.mask_key: labelmask_ds}
                                                 )
                                      + Pad(label.mask_key, Coordinate(output_size)))
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
            cropno=crop["number"], file=crop["parent"], ds=raw_ds, ak=ak_raw))
        raw_src = (
            ZarrSource(
                crop["parent"],
                {ak_raw: raw_ds},
                array_specs={ak_raw: ArraySpec(voxel_size=voxel_size_input)})
            + Pad(ak_raw, Coordinate(input_size), 0)
        )
        all_srcs.append(raw_src)

        # add gt mask source
        logging.debug("Adding ZarrSource ({file:}/{ds:}) for crop {cropno:}, providing {ak}".format(
            cropno=crop["number"], file=crop["parent"], ds=mask_ds, ak=ak_mask))
        mask_src = (
            ZarrSource(
                crop["parent"],
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
        contr_adj = n5file["volumes/raw"].attrs["contrastAdjustment"]
        scale = 255.0 / (float(contr_adj["max"]) - float(contr_adj["min"]))
        shift = - scale * float(contr_adj["min"])
        logging.debug("Adjusting contrast with scale {scale:} and shift {shift:}".format(scale=scale, shift=shift))
        crop_src += IntensityScaleShift(ak_raw,
                                        scale,
                                        shift
                                        )

        return crop_src

    def network_setup():

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

    ak_raw = ArrayKey("RAW")
    ak_labels = ArrayKey("GT_LABELS")
    ak_labels_downsampled = ArrayKey("GT_LABELS_DOWNSAMPLED")
    ak_mask = ArrayKey("MASK")
    ak_labelmasks_comb = ArrayKey("LABELMASKS_COMBINED")
    input_size = Coordinate(input_shape) * voxel_size_input
    output_size = Coordinate(output_shape) * voxel_size

    keep_thr = float(min_masked_voxels)/np.prod(output_shape)
    one_vx_thr = 1./np.prod(output_shape)

    client = pymongo.MongoClient("cosem.int.janelia.org:27017", username=db_username, password=db_password)
    db = client[db_name]  # db_name = "crops"
    collection = db[gt_version]  # gt_version = "v0003"
    filter = {"completion": {"$gte": completion_min}}
    skip = {"_id": 0, "number": 1, "labels": 1, "parent": 1, "dimensions": 1}

    net_io_names, start_iteration, inputs, outputs = network_setup()

    # construct batch request
    request = BatchRequest()
    #if len(label_filter(lambda l: not l.separate_labelset)) > 0:
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
    if len(label_filter(lambda l: not l.separate_labelset)) > 0:
        snapshot_data[ak_labels] = "volumes/labels/gt_labels"
    for label in label_filter(lambda l: l.separate_labelset):
        snapshot_data[label.gt_key] = "volumes/labels/gt_"+label.labelname
    for label in labels:
        snapshot_data[label.gt_dist_key] = "volumes/labels/gt_dist_" + label.labelname
        snapshot_data[label.pred_dist_key] = "volumes/labels/pred_dist_" + label.labelname
        snapshot_data[label.mask_key] = "volumes/masks/" +label.labelname

    # specify snapshot request
    snapshot_request = BatchRequest()

    crop_srcs = []
    crop_sizes = []
    for crop in collection.find(filter, skip):
        if len(set(get_all_annotated_label_ids(crop)).intersection(set(get_all_labelids(labels)))) > 0:
            logging.info("Adding crop number {0:}".format(crop["number"]))
            if voxel_size_input != voxel_size:
                for subsample_variant in range(int(np.prod(voxel_size_input/voxel_size))):
                    crop_srcs.append(make_crop_source(crop, subsample_variant))
                    crop_sizes.append(get_crop_size(crop))
            else:
                crop_srcs.append(make_crop_source(crop))
                crop_sizes.append(get_crop_size(crop))

    pipeline = (tuple(crop_srcs)
                + RandomProvider(crop_sizes)
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
            max_distance=2.76 * dt_scaling_factor,
        )

    # combine distances for centrosomes

    centrosome = get_label("centrosome")
    microtubules = get_label("microtubules")
    microtubules_out = get_label("microtubules_out")
    subdistal_app = get_label("subdistal_app")
    distal_app = get_label("distal_app")

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

    for label in labels:
        pipeline += TanhSaturate(label.gt_dist_key, dt_scaling_factor)

    for label in label_filter(lambda l: l.scale_loss):
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
