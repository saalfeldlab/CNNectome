from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddDistance
import tensorflow as tf
import os
import math
import json
import sys
import numpy as np

print("syspath", sys.path)
import z5py


def train_until(max_iteration, data_sources, input_shape, output_shape):
    ArrayKey('GT_RAW')
    ArrayKey('INPUT_RAW')
    ArrayKey('PRED_RAW')
    data_providers = []
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/superresolution/{0:}.n5"
    voxel_size_up = Coordinate((4, 4, 4))
    voxel_size_down = Coordinate((8, 8, 8))
    input_size = Coordinate(input_shape) * voxel_size_down
    output_size = Coordinate(output_shape) * voxel_size_up
    shift = np.array(input_size)-np.array(output_size)
    assert (shift % 2).all() == 0
    shift /= 2
    shift_pct = tuple(shift/np.array(input_size))

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    request = BatchRequest()
    request.add(ArrayKeys.INPUT_RAW, input_size, voxel_size=voxel_size_down)
    request.add(ArrayKeys.GT_RAW, input_size, voxel_size=voxel_size_up)
    #snapshot_request =x BatchRequest()
    #snapshot_request.add(ArrayKeys.PRED_RAW, output_size, voxel_size=voxel_size)

    # load latest ckpt for weights if available
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')

    # construct DAG
    for src in data_sources:
        n5_source = N5Source(
            data_dir.format(src),
            datasets={
                ArrayKeys.GT_RAW: 'volumes/raw'
            }
        )
        data_providers.append(n5_source)

    data_sources = tuple(
        provider +
        Normalize(ArrayKeys.GT_RAW) +
        Pad(ArrayKeys.GT_RAW, Coordinate((400, 400, 400))) +
        RandomLocation()
        for provider in data_providers
    )

    train_pipeline_part1 = (
        data_sources +
        ElasticAugment((100, 100, 100), (10., 10., 10.), (0, math.pi / 2.0),
                       prob_slip=0, prob_shift=0, max_misalign=0,
                       subsample=8) +
        SimpleAugment() +
        ElasticAugment((40, 1000, 1000), (10., 0., 0.), (0, 0), subsample=8) +
        IntensityAugment(ArrayKeys.GT_RAW, 0.9, 1.1, -0.1, 0.1) +
        IntensityScaleShift(ArrayKeys.GT_RAW, 2, -1) +
        ZeroOutConstSections(ArrayKeys.GT_RAW) +
        CopyArray(ArrayKeys.GT_RAW, ArrayKeys.INPUT_RAW))

    train_pipeline = (train_pipeline_part1 +
        Crop(ArrayKeys.GT_RAW, fraction_negative=shift_pct, fraction_positive=shift_pct) +
        DownSample(ArrayKeys.INPUT_RAW)+
        NoiseAugment(ArrayKeys.INPUT_RAW, )+
        PreCache(cache_size=40, num_workers=10) +
        Train('build',
              optimizer=net_io_names['optimizer'],
              loss=net_io_names['loss'],
              inputs={
                  net_io_names['raw']: ArrayKeys.RAW
              },
              summary=net_io_names['summary'],
              log_dir='log',
              outputs={
                  net_io_names['pred_raw']: ArrayKeys.PRED_RAW
              },
              gradients={}
              ) +
        Snapshot({ArrayKeys.RAW: 'volumes/raw', ArrayKeys.PRED_RAW: 'volumes/pred_raw'},
                 every=500,
                 output_filename='batch_{iteration}.hdf',
                 output_dir='snapshots/',
                 additional_request=snapshot_request) +
        PrintProfilingStats(every=50)
    )
    # no intensity augment cause currently can't apply the same to both in and out



    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    set_verbose(False)
    data_sources = ['block1_4nm']
    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    max_iteration = 400000
    train_until(max_iteration, data_sources, input_shape, output_shape)
