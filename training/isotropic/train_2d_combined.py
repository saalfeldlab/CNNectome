from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddBoundaryDistance
import tensorflow as tf
import malis
import os
import math
import json
import h5py


def train_until(max_iteration, data_dir, data_sources, input_shape, output_shape, loss_name):
    ArrayKey('RAW_R')
    ArrayKey('RAW_G')
    ArrayKey('RAW_B')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')
    ArrayKey('GT_DIST_SCALE')
    ArrayKey('GT_AFF_SCALE')
    ArrayKey('LOSS_GRADIENT')
    ArrayKey('GT_DIST')
    ArrayKey('PREDICTED_DIST')
    ArrayKey('GT_AFF')
    ArrayKey('PREDICTED_AFF1')
    ArrayKey('PREDICTED_AFF3')
    ArrayKey('PREDICTED_AFF9')

    data_providers = []
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')

    for sample in data_sources:
        h5_source = Hdf5Source(
            data_dir,
            datasets={
                ArrayKeys.RAW_R: sample + '/image-r',
                ArrayKeys.RAW_G: sample + '/image-g',
                ArrayKeys.RAW_B: sample + '/image-b',
                ArrayKeys.GT_LABELS: sample+'/mask',
            }
        )

        data_providers.append(h5_source)

    #todo: dvid source

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((1, 1))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    # input_size = Coordinate((132,)*3) * voxel_size
    # output_size = Coordinate((44,)*3) * voxel_size

    # specifiy which volumes should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW_R, input_size)
    request.add(ArrayKeys.RAW_G, input_size)
    request.add(ArrayKeys.RAW_B, input_size)
    request.add(ArrayKeys.GT_LABELS, output_size)
    request.add(ArrayKeys.GT_AFF, output_size)
    request.add(ArrayKeys.GT_DIST, output_size)
    request.add(ArrayKeys.GT_DIST_SCALE, output_size)
    request.add(ArrayKeys.GT_AFF_SCALE, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(ArrayKeys.RAW_R) + # ensures RAW is in float in [0, 1]
        Normalize(ArrayKeys.RAW_G) +
        Normalize(ArrayKeys.RAW_B) +
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW_R, Coordinate((200, 200)) * voxel_size) +
        Pad(ArrayKeys.RAW_G, Coordinate((200, 200)) * voxel_size) +
        Pad(ArrayKeys.RAW_B, Coordinate((200, 200)) * voxel_size) +
        RandomLocation() + # chose a random location inside the provided arrays
        Reject(ArrayKeys.GT_LABELS, min_masked=0.0, reject_probability=0.95)

        for provider in data_providers)

    snapshot_request = BatchRequest({
        ArrayKeys.LOSS_GRADIENT:      request[ArrayKeys.GT_LABELS],
        ArrayKeys.PREDICTED_DIST:     request[ArrayKeys.GT_DIST],
        ArrayKeys.PREDICTED_AFF1:      request[ArrayKeys.GT_AFF],
        ArrayKeys.PREDICTED_AFF3:      request[ArrayKeys.GT_AFF],
        ArrayKeys.PREDICTED_AFF9:      request[ArrayKeys.GT_AFF],
        ArrayKeys.LOSS_GRADIENT:      request[ArrayKeys.GT_DIST],

    })


    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment((40, 40), (2., 2.), (0, math.pi/2.0),
                       subsample=4, spatial_dims=2) +
        SimpleAugment() +
        IntensityAugment(ArrayKeys.RAW_R, 0.9, 1.1, -0.1, 0.1) +
        IntensityAugment(ArrayKeys.RAW_G, 0.9, 1.1, -0.1, 0.1) +
        IntensityAugment(ArrayKeys.RAW_B, 0.9, 1.1, -0.1, 0.1) +
        DefectAugment(ArrayKeys.RAW_R, prob_low_contrast=0.01, contrast_scale=0.5) +
        DefectAugment(ArrayKeys.RAW_G, prob_low_contrast=0.01, contrast_scale=0.5) +
        DefectAugment(ArrayKeys.RAW_B, prob_low_contrast=0.01, contrast_scale=0.5) +
        IntensityScaleShift(ArrayKeys.RAW_R, 2, -1) +
        IntensityScaleShift(ArrayKeys.RAW_G, 2, -1) +
        IntensityScaleShift(ArrayKeys.RAW_B, 2, -1) +
        ZeroOutConstSections(ArrayKeys.RAW_R) +
        ZeroOutConstSections(ArrayKeys.RAW_G) +
        ZeroOutConstSections(ArrayKeys.RAW_B) +
        AddBoundaryDistance(label_array_key=ArrayKeys.GT_LABELS,
                            distance_array_key=ArrayKeys.GT_DIST,
                            normalize='tanh',
                            normalize_args=150) +
        AddAffinities([[-1, 0], [0, -1],
                       [-3, 0], [0, -3],
                       [-9, 0], [0, -9]],
                      ArrayKeys.GT_LABELS,
                      ArrayKeys.GT_AFF) +
        BalanceLabels(ArrayKeys.GT_AFF, ArrayKeys.GT_AFF_SCALE) +
        BalanceByThreshold(labels=ArrayKeys.GT_LABELS, scales=ArrayKeys.GT_DIST_SCALE) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names['raw_r']:     ArrayKeys.RAW_R,
                net_io_names['raw_g']:     ArrayKeys.RAW_G,
                net_io_names['raw_b']:     ArrayKeys.RAW_B,
                net_io_names['gt_dist']: ArrayKeys.GT_DIST,
                net_io_names['gt_aff']:  ArrayKeys.GT_AFF,
                net_io_names['loss_weights_dist']: ArrayKeys.GT_DIST_SCALE,
                net_io_names['loss_weights_aff']: ArrayKeys.GT_AFF_SCALE
            },
            summary=net_io_names['summary'],
            log_dir='log',
            outputs={
                net_io_names['dist']:  ArrayKeys.PREDICTED_DIST,
                net_io_names['aff1']:  ArrayKeys.PREDICTED_AFF1,
                net_io_names['aff3']:  ArrayKeys.PREDICTED_AFF3,
                net_io_names['aff9']:  ArrayKeys.PREDICTED_AFF9
            },
            gradients={
                net_io_names['dist']: ArrayKeys.LOSS_GRADIENT
            }) +
        Snapshot({
                ArrayKeys.RAW_R:          'volumes/raw_r',
                ArrayKeys.RAW_G:          'volumes/raw_g',
                ArrayKeys.RAW_B:          'volumes/raw_b',
                ArrayKeys.GT_DIST:        'volumes/labels/dist',
                ArrayKeys.GT_AFF:         'volumes/labels/aff',
                ArrayKeys.GT_LABELS:      'volumes/labels/nuclei',
                ArrayKeys.PREDICTED_DIST: 'volumes/predictions/dist',
                ArrayKeys.PREDICTED_AFF1: 'volumes/predictions/aff1',
                ArrayKeys.PREDICTED_AFF3: 'volumes/predictions/aff3',
                ArrayKeys.PREDICTED_AFF9: 'volumes/predictions/aff9',
                ArrayKeys.LOSS_GRADIENT:  'volumes/loss_gradient',
            },
            every=500,
            output_filename='batch_{iteration}.hdf',
            output_dir='snapshots/',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=50))

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    set_verbose(False)
    data_dir = "/groups/saalfeld/saalfeldlab/kaggle-nuclei/stage1_train.rgb.hdf5"
    hf = h5py.File(data_dir, 'r')
    data_sources = [k for k in hf.keys() if k!='__DATA_TYPES__']
    input_shape = (256, 256)
    output_shape = (29, 29)
    max_iteration = 400000
    loss_name = 'loss_total'
    train_until(max_iteration, data_dir, data_sources, input_shape, output_shape, loss_name)
