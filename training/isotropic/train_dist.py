from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddBoundaryDistance
import tensorflow as tf
import os
import math
import json


def train_until(max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name):
    ArrayKey('RAW')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')

    ArrayKey('GT_SCALE')
    ArrayKey('LOSS_GRADIENT')
    ArrayKey('GT_DIST')
    ArrayKey('PREDICTED_DIST')

    data_providers = []
    fib25_dir = "/groups/saalfeld/saalfeldlab/larissa/data/gunpowder/fib25/"
    if 'fib25h5' in data_sources:

        for volume_name in ("tstvol-520-1", "tstvol-520-2", "trvol-250-1", "trvol-250-2"):
            h5_source = Hdf5Source(os.path.join(fib25_dir, volume_name+'.hdf'),
                                   datasets={ArrayKeys.RAW: 'volumes/raw',
                                             ArrayKeys.GT_LABELS:'volumes/labels/clefts',
                                             ArrayKeys.GT_MASK: 'volumes/masks/groundtruth'},
                                   volume_specs={
                                       Array.GT_MASK: ArraySpec(interpolatable=False)
                                   }
            )
            data_providers.append(h5_source)

    fib19_dir = "/groups/saalfeld/saalfeldlab/larissa/fib19"
    #if 'fib19h5' in data_sources:
    #    for volume_name in ("trvol-250", "trvol-600"):
    #        h5_source = prepare_h5source(fib19_dir, volume_name)
    #        data_providers.append(h5_source)

    #todo: dvid source

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate((196,)*3) * voxel_size
    output_size = Coordinate((92,)*3) * voxel_size
    # input_size = Coordinate((132,)*3) * voxel_size
    # output_size = Coordinate((44,)*3) * voxel_size

    # specifiy which volumes should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    request.add(ArrayKeys.GT_LABELS, output_size)
    request.add(ArrayKeys.GT_MASK, output_size)
    #request.add(VolumeTypes.GT_SCALE, output_size)
    request.add(ArrayKeys.GT_DIST, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize() + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW, Coordinate((100, 100, 100)) * voxel_size)+
        Pad(ArrayKeys.GT_MASK, Coordinate((100, 100, 100)) * voxel_size)+

        RandomLocation() + # chose a random location inside the provided volumes
        Reject() # reject batches wich do contain less than 50% labelled data
        for provider in data_providers)

    snapshot_request = BatchRequest({
        ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_LABELS],
        ArrayKeys.PREDICTED_DIST: request[ArrayKeys.GT_LABELS],
        ArrayKeys.LOSS_GRADIENT: request[ArrayKeys.GT_DIST],

    })

    #artifact_source = (
    #    Hdf5Source(
    #        os.path.join(data_dir, 'sample_ABC_padded_20160501.defects.hdf'),
    #        datasets = {
    #            VolumeTypes.RAW: 'defect_sections/raw',
    #            VolumeTypes.ALPHA_MASK: 'defect_sections/mask',
    #        },
    #        volume_specs = {
    #            VolumeTypes.RAW: VolumeSpec(voxel_size=(40, 4, 4)),
    #            VolumeTypes.ALPHA_MASK: VolumeSpec(voxel_size=(40, 4, 4)),
    #        }
    #    ) +
    #    RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
    #    Normalize() +
    #    IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
    #    ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], subsample=8) +
    #    SimpleAugment(transpose_only_xy=True)
    #)

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment([40,40,40], [2,2,2], [0,math.pi/2.0], prob_slip=0.01,prob_shift=0.05,max_misalign=1,
                       subsample=8) +
        SimpleAugment() +
        IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1) +
        IntensityScaleShift(ArrayKeys.RAW, 2, -1) +
        ZeroOutConstSections(ArrayKeys.RAW) +
        GrowBoundary(steps=1) +
        #SplitAndRenumberSegmentationLabels() +
        #AddGtAffinities(malis.mknhood3d()) +
        AddBoundaryDistance(label_volume_type=ArrayKeys.GT_LABELS,
                            distance_volume_type=ArrayKeys.GT_DIST,
                            normalize='tanh',
                            normalize_args=dt_scaling_factor
                            )+
        BalanceLabels(ArrayKeys.GT_LABELs, ArrayKeys.GT_SCALE, ArrayKeys.GT_MASK)+
        #BalanceByThreshold(
        #    labels=VolumeTypes.GT_DIST,
        #    scales= VolumeTypes.GT_SCALE) +
            # {
            #     VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_SCALE
            # },
            # {
            #     VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_MASK
            # }) +
        PreCache(
            cache_size=40,
            num_workers=10)+
        #DefectAugment(
        #    prob_missing=0.03,
        #    prob_low_contrast=0.01,
        #    prob_artifact=0.03,
        #    artifact_source=artifact_source,
        #    contrast_scale=0.5) +
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names['raw']: ArrayKeys.RAW,
                net_io_names['gt_dist']: ArrayKeys.GT_DIST,
                #net_io_names['loss_weights']: VolumeTypes.GT_SCALE
            },
            summary=net_io_names['summary'],
            log_dir='log',
            outputs={
                net_io_names['dist']: ArrayKeys.PREDICTED_DIST
            },
            gradients={
                net_io_names['dist']: ArrayKeys.LOSS_GRADIENT
            }) +
        Snapshot({
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids',
                ArrayKeys.GT_DIST: 'volumes/labels/distances',
                ArrayKeys.PREDICTED_DIST: 'volumes/labels/pred_distances',
                ArrayKeys.LOSS_GRADIENT: 'volumes/loss_gradient',
            },
            every=1000,
            output_filename='batch_{iteration}.hdf',
            output_dir='snapshots/',
            additional_request=snapshot_request) +

        PrintProfilingStats(every=10)
     )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    data_sources = ['fib25h5']#, 'fib19h5']
    max_iteration = 400000
    dt_scaling_factor = 50
    input_shape = (430,430,430)
    output_shape = (218,218,218)
    loss_name = 'loss_balanced_syn'
    train_until(max_iteration, data_sources, input_Shape, output_shape, dt_scaling_factor, loss_name)