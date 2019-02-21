from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddBoundaryDistance, AddDistance, AddPrePostCleftDistance
import tensorflow as tf
import os
import math
import json
import csv
import logging

def make_cleft_to_prepostsyn_neuron_id_dict(csv_files):
    cleft_to_pre = dict()
    cleft_to_post = dict()
    for csv_f in csv_files:
        f = open(csv_f, 'r')
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            if int(row[10]) >= 0:
                try:
                    cleft_to_pre[int(row[10])].add(int(row[0]))
                except KeyError:
                    cleft_to_pre[int(row[10])] = {int(row[0])}
                try:
                    cleft_to_post[int(row[10])].add(int(row[5]))
                except KeyError:
                    cleft_to_post[int(row[10])] = {int(row[5])}
    return cleft_to_pre, cleft_to_post


def train_until(max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, cremi_version,
                aligned):
    ArrayKey('RAW')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')
    ArrayKey('GT_CLEFTS')
    ArrayKey('GT_MASK')
    ArrayKey('TRAINING_MASK')
    ArrayKey('CLEFT_SCALE')
    ArrayKey('PRE_SCALE')
    ArrayKey('POST_SCALE')
    ArrayKey('LOSS_GRADIENT')
    ArrayKey('GT_CLEFT_DIST')
    ArrayKey('PRED_CLEFT_DIST')
    ArrayKey('GT_PRE_DIST')
    ArrayKey('PRED_PRE_DIST')
    ArrayKey('GT_POST_DIST')
    ArrayKey('PRED_POST_DIST')
    ArrayKey('GT_POST_DIST')
    data_providers = []
    if cremi_version == '2016':
        cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2016/"
        filename = 'sample_{0:}_padded_20160501.'
    elif cremi_version == '2017':
        cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/"
        filename = 'sample_{0:}_padded_20170424.'
    if aligned:
        filename += 'aligned.'
    filename += '0bg.hdf'
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')
    for sample in data_sources:
        print(sample)
        h5_source = Hdf5Source(
            os.path.join(cremi_dir, filename.format(sample)),
            datasets={
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_CLEFTS: 'volumes/labels/clefts',
                ArrayKeys.GT_MASK: 'volumes/masks/groundtruth',
                ArrayKeys.TRAINING_MASK: 'volumes/masks/validation',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids'
            },
            array_specs={
                ArrayKeys.GT_MASK: ArraySpec(interpolatable=False),
                ArrayKeys.GT_CLEFTS: ArraySpec(interpolatable=False),
                ArrayKeys.TRAINING_MASK: ArraySpec(interpolatable=False)
            }
        )
        data_providers.append(h5_source)

    if cremi_version == '2017':
        csv_files = [os.path.join(cremi_dir, 'cleft-partners_' + sample + '_2017.csv') for sample in data_sources]
    elif cremi_version == '2016':
        csv_files = [os.path.join(cremi_dir, 'cleft-partners-' + sample + '-20160501.aligned.corrected.csv') for
                     sample in data_sources]
    cleft_to_pre, cleft_to_post = make_cleft_to_prepostsyn_neuron_id_dict(csv_files)
    print(cleft_to_pre, cleft_to_post)
    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = input_size - output_size
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    request.add(ArrayKeys.GT_LABELS, output_size)
    request.add(ArrayKeys.GT_CLEFTS, output_size)
    request.add(ArrayKeys.GT_MASK, output_size)
    request.add(ArrayKeys.TRAINING_MASK,output_size)
    request.add(ArrayKeys.CLEFT_SCALE, output_size)
    request.add(ArrayKeys.GT_CLEFT_DIST, output_size)
    request.add(ArrayKeys.GT_PRE_DIST, output_size)
    request.add(ArrayKeys.GT_POST_DIST, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(ArrayKeys.RAW) + # ensures RAW is in float in [0, 1]
        IntensityScaleShift(ArrayKeys.TRAINING_MASK, -1, 1) +
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW, None) +
        Pad(ArrayKeys.GT_MASK, None) +
        Pad(ArrayKeys.TRAINING_MASK, context) +
        RandomLocation(min_masked=0.99, mask=ArrayKeys.TRAINING_MASK) + # chose a random location inside the provided arrays
        Reject(ArrayKeys.GT_MASK)+ # reject batches which do contain less than 50% labelled data
        Reject(ArrayKeys.GT_CLEFTS, min_masked=0.0, reject_probability=0.95)
        for provider in data_providers)

    snapshot_request = BatchRequest({
        ArrayKeys.LOSS_GRADIENT:         request[ArrayKeys.GT_CLEFTS],
        ArrayKeys.PRED_CLEFT_DIST:       request[ArrayKeys.GT_CLEFT_DIST],
        ArrayKeys.PRED_PRE_DIST:         request[ArrayKeys.GT_PRE_DIST],
        ArrayKeys.PRED_POST_DIST:        request[ArrayKeys.GT_POST_DIST],
    })

    artifact_source = (
        Hdf5Source(
            os.path.join(cremi_dir, 'sample_ABC_padded_20160501.defects.hdf'),
            datasets={
                ArrayKeys.RAW:        'defect_sections/raw',
                ArrayKeys.ALPHA_MASK: 'defect_sections/mask',
            },
            array_specs={
                ArrayKeys.RAW:        ArraySpec(voxel_size=(40, 4, 4)),
                ArrayKeys.ALPHA_MASK: ArraySpec(voxel_size=(40, 4, 4)),
            }
        ) +
        RandomLocation(min_masked=0.05, mask=ArrayKeys.ALPHA_MASK) +
        Normalize(ArrayKeys.RAW) +
        IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi/2.0), subsample=8) +
        SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment((4, 40, 40), (0., 2., 2.), (0, math.pi/2.0),
                       prob_slip=0.05, prob_shift=0.05, max_misalign=10,
                       subsample=8) +
        SimpleAugment(transpose_only=[1,2], mirror_only=[1,2]) +
        IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(ArrayKeys.RAW,
                      prob_missing=0.03,
                      prob_low_contrast=0.01,
                      prob_artifact=0.03,
                      artifact_source=artifact_source,
                      artifacts=ArrayKeys.RAW,
                      artifacts_mask=ArrayKeys.ALPHA_MASK,
                      contrast_scale=0.5) +
        IntensityScaleShift(ArrayKeys.RAW, 2, -1) +
        ZeroOutConstSections(ArrayKeys.RAW) +

        AddDistance(label_array_key=ArrayKeys.GT_CLEFTS,
                    distance_array_key=ArrayKeys.GT_CLEFT_DIST,
                    normalize='tanh',
                    normalize_args=dt_scaling_factor
                    ) +
        AddPrePostCleftDistance(ArrayKeys.GT_CLEFTS,
                                ArrayKeys.GT_LABELS,
                                ArrayKeys.GT_PRE_DIST,
                                ArrayKeys.GT_POST_DIST,
                                cleft_to_pre,
                                cleft_to_post,
                                normalize='tanh',
                                normalize_args=dt_scaling_factor,
                                include_cleft=False
                                )+
        BalanceByThreshold(labels=ArrayKeys.GT_CLEFT_DIST,
                           scales=ArrayKeys.CLEFT_SCALE,
                           mask=ArrayKeys.GT_MASK) +
        BalanceByThreshold(labels=ArrayKeys.GT_PRE_DIST,
                           scales=ArrayKeys.PRE_SCALE,
                           mask=ArrayKeys.GT_MASK,
                           threshold=-0.5) +
        BalanceByThreshold(labels=ArrayKeys.GT_POST_DIST,
                           scales=ArrayKeys.POST_SCALE,
                           mask=ArrayKeys.GT_MASK,
                           threshold=-0.5) +
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names[loss_name],
            inputs={
                net_io_names['raw']: ArrayKeys.RAW,
                net_io_names['gt_cleft_dist']: ArrayKeys.GT_CLEFT_DIST,
                net_io_names['gt_pre_dist']: ArrayKeys.GT_PRE_DIST,
                net_io_names['gt_post_dist']: ArrayKeys.GT_POST_DIST,
                net_io_names['loss_weights_cleft']: ArrayKeys.CLEFT_SCALE,
                net_io_names['loss_weights_pre']: ArrayKeys.CLEFT_SCALE,
                net_io_names['loss_weights_post']: ArrayKeys.CLEFT_SCALE,
                net_io_names['mask']: ArrayKeys.GT_MASK
            },
            summary=net_io_names['summary'],
            log_dir='log',
            outputs={
                net_io_names['cleft_dist']: ArrayKeys.PRED_CLEFT_DIST,
                net_io_names['pre_dist']: ArrayKeys.PRED_PRE_DIST,
                net_io_names['post_dist']: ArrayKeys.PRED_POST_DIST
            },
            gradients={
                net_io_names['cleft_dist']: ArrayKeys.LOSS_GRADIENT
            }) +
        Snapshot({
            ArrayKeys.RAW:             'volumes/raw',
            ArrayKeys.GT_CLEFTS:       'volumes/labels/gt_clefts',
            ArrayKeys.GT_CLEFT_DIST:   'volumes/labels/gt_clefts_dist',
            ArrayKeys.PRED_CLEFT_DIST: 'volumes/labels/pred_clefts_dist',
            ArrayKeys.LOSS_GRADIENT:   'volumes/loss_gradient',
            ArrayKeys.PRED_PRE_DIST:   'volumes/labels/pred_pre_dist',
            ArrayKeys.PRED_POST_DIST:  'volumes/labels/pred_post_dist',
            ArrayKeys.GT_PRE_DIST:     'volumes/labels/gt_pre_dist',
            ArrayKeys.GT_POST_DIST:    'volumes/labels/gt_post_dist'
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
    logging.basicConfig(level=logging.INFO)
    data_sources = ['A', 'B', 'C']#, 'B', 'C']
    input_shape = (43, 430, 430)
    output_shape = (23, 218, 218)
    dt_scaling_factor = 50
    max_iteration = 400000
    loss_name = 'loss_total'
    cremi_version = '2017'
    aligned = True
    train_until(max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, cremi_version,
                aligned)
