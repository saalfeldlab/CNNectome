from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import sys
import logging
print("syspath", sys.path)
import z5py

class Label(object):
    def __init__(self, labelname, labelid, scale_loss=True, scale_key=None):
        self.labelname= labelname
        self.labelid = labelid

        self.gt_dist_key = ArrayKey('GT_DIST_'+self.labelname.upper())
        self.pred_dist_key = ArrayKey('PRED_DIST_'+self.labelname.upper())
        self.scale_loss = scale_loss
        if self.scale_loss:
            self.scale_key = ArrayKey('SCALE_'+self.labelname.upper())
        if scale_key is not None:
            self.scale_key = scale_key


def train_until(max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, labels):
    ArrayKey('RAW')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')
    ArrayKey('MASK')


    data_providers = []
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5"
    voxel_size = Coordinate((2, 2, 2))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')
    for src in data_sources:
        n5_source = N5Source(
            os.path.join(data_dir.format(src)),
            datasets={
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/all',
                ArrayKeys.MASK: 'volumes/mask'
            },
            array_specs={
                ArrayKeys.MASK: ArraySpec(interpolatable=False)
            }
        )
        data_providers.append(n5_source)

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)


    inputs = dict()
    inputs[net_io_names['raw']] = ArrayKeys.RAW
    outputs = dict()
    snapshot = dict()
    snapshot[ArrayKeys.RAW] = 'volumes/raw'
    snapshot[ArrayKeys.GT_LABELS] = 'volumes/labels/gt_labels'
    for label in labels:
        inputs[net_io_names['gt_'+label.labelname]] = label.gt_dist_key
        if label.scale_loss or label.scale_key is not None:
            inputs[net_io_names['w_'+label.labelname]] = label.scale_key
        outputs[net_io_names[label.labelname]] = label.pred_dist_key
        snapshot[label.gt_dist_key] = 'volumes/labels/gt_dist_'+label.labelname
        snapshot[label.pred_dist_key] = 'volumes/labels/pred_dist_'+label.labelname

    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    snapshot_request = BatchRequest()

    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size)
    request.add(ArrayKeys.GT_LABELS, output_size,  voxel_size=voxel_size)
    request.add(ArrayKeys.MASK, output_size, voxel_size=voxel_size)

    for label in labels:
        request.add(label.gt_dist_key, output_size, voxel_size=voxel_size)
        snapshot_request.add(label.pred_dist_key, output_size, voxel_size=voxel_size)
        if label.scale_loss:
            request.add(label.scale_key, output_size, voxel_size=voxel_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(ArrayKeys.RAW) + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW, None) +
        RandomLocation(min_masked=0.5, mask=ArrayKeys.MASK) # chose a random location inside the provided arrays
        #Reject(ArrayKeys.MASK) # reject batches wich do contain less than 50% labelled data

        for provider in data_providers)

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment((100, 100, 100), (10., 10., 10.), (0, math.pi/2.0),
                       prob_slip=0, prob_shift=0, max_misalign=0,
                       subsample=8) +
        SimpleAugment() +
        #ElasticAugment((40, 1000, 1000), (10., 0., 0.), (0, 0), subsample=8) +
        IntensityAugment(ArrayKeys.RAW, 0.95, 1.05, -0.05, 0.05) +
        IntensityScaleShift(ArrayKeys.RAW, 2, -1) +
        ZeroOutConstSections(ArrayKeys.RAW))

    for label in labels:
        train_pipeline += AddDistance(label_array_key=ArrayKeys.GT_LABELS,
                        distance_array_key=label.gt_dist_key,
                        normalize='tanh',
                        normalize_args=dt_scaling_factor,
                        label_id=label.labelid)

    train_pipeline = (train_pipeline)
    for label in labels:
        if label.scale_loss:
            train_pipeline += BalanceByThreshold(label.gt_dist_key, label.scale_key, mask=ArrayKeys.MASK)
    train_pipeline = (
        train_pipeline +
        PreCache(
            cache_size=40,
            num_workers=10)+

        Train(
            'build',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names[loss_name],
            inputs=inputs,
            summary=net_io_names['summary'],
            log_dir='log',
            outputs=outputs,
            gradients={}
        ) +
        Snapshot(snapshot,
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
    #set_verbose(False)
    logging.basicConfig(level=logging.INFO)
    data_sources = ['gt_cell2_v1', ]
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    dt_scaling_factor = 50
    max_iteration = 400000
    loss_name = 'loss_total_unbalanced'
    labels = []
    labels.append(Label('ECS', (6,7)))
    labels.append(Label('cell', (1,2,3,4,5,8,9,10,11,12,13,14)))
    labels.append(Label('plasma_membrane', 5))
    labels.append(Label('ERES', (12,13)))
    labels.append(Label('ERES_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('MVB', (3,9)))
    labels.append(Label('MVB_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('er', (4,8)))
    labels.append(Label('er_membrane', 4, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('mito', (1,2)))
    labels.append(Label('mito_membrane', 2, scale_loss=False, scale_key=labels[-1].scale_key))
    labels.append(Label('vesicles', 10))
    labels.append(Label('microtubules', 11))
    train_until(max_iteration, data_sources, input_shape, output_shape, dt_scaling_factor, loss_name, labels)
