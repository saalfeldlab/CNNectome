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
import collections
print("syspath", sys.path)
import z5py


class Label(object):
    def __init__(self, labelname, labelid, scale_loss=True, scale_key=None,
                 data_dir="/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5",
                 data_sources=None):

        self.labelname= labelname
        if not isinstance(labelid, collections.Iterable):
            labelid = (labelid, )
        self.labelid = labelid
        self.gt_dist_key = ArrayKey('GT_DIST_'+self.labelname.upper())
        self.pred_dist_key = ArrayKey('PRED_DIST_'+self.labelname.upper())
        #self.mask_key = ArrayKey('MASK_'+self.labelname.upper())
        self.scale_loss = scale_loss
        self.data_dir = data_dir
        self.data_sources = data_sources
        self.total_voxels = compute_total_voxels(self.data_dir, self.data_sources)
        num = 0
        for ds in data_sources:
            zf = z5py.File(data_dir.format(ds), use_zarr_format=False)
            for l in labelid:
                if l in zf['volumes/labels/all'].attrs['relabeled_ids']:
                    num += zf['volumes/labels/all'].attrs['relabeled_counts'][zf['volumes/labels/all'].attrs[
                        'relabeled_ids'].index(l)]
        if num > 0:
            self.class_weight = float(self.total_voxels) / num
        else:
            self.class_weight = 0.
        print(labelname, self.class_weight)
        if self.scale_loss:
            self.scale_key = ArrayKey('SCALE_'+self.labelname.upper())
        if scale_key is not None:
            self.scale_key = scale_key


def compute_total_voxels(data_dir, data_sources):
    voxels = 0
    for ds in data_sources:
        zf = z5py.File(data_dir.format(ds), use_zarr_format=False)
        for c in zf['volumes/labels/all'].attrs['orig_counts']:
            voxels += c
    return voxels


def train_until(max_iteration, data_sources,  labeled_voxels,ribo_sources, input_shape, output_shape, \
                                                          dt_scaling_factor, loss_name,
                labels):
    ArrayKey('RAW')
    ArrayKey('RAW_UP')
    ArrayKey('ALPHA_MASK')
    ArrayKey('GT_LABELS')
    ArrayKey('MASK')
    ArrayKey('RIBO_GT')
    ArrayKey('RIBO_MASK')
    ArrayKey('RIBO_MASK_UP')
    ArrayKey('MASK_UP')

    data_providers = []

    voxel_size_up = Coordinate((2, 2, 2))
    voxel_size_orig = Coordinate((4, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size_orig
    output_size = Coordinate(output_shape) * voxel_size_orig
    context = input_size-output_size
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')
    for src in data_sources:
        if src not in ribo_sources:
            n5_source = N5Source(
                os.path.join(data_dir.format(src)),
                datasets={
                    ArrayKeys.RAW_UP: 'volumes/raw',
                    ArrayKeys.GT_LABELS: 'volumes/labels/all',
                    ArrayKeys.MASK_UP: 'volumes/mask',
                    ArrayKeys.RIBO_GT: 'volumes/ribosomes_mask',
                    ArrayKeys.RIBO_MASK_UP: 'volumes/ribosomes_mask'
                },
                array_specs={
                    ArrayKeys.MASK_UP: ArraySpec(interpolatable=False),
                    ArrayKeys.RIBO_MASK_UP: ArraySpec(interpolatable=False)
                }
        )
        else:
            n5_source = N5Source(
                os.path.join(data_dir.format(src)),
                datasets={
                    ArrayKeys.RAW_UP: 'volumes/raw',
                    ArrayKeys.GT_LABELS: 'volumes/labels/all',
                    ArrayKeys.MASK_UP: 'volumes/mask',
                    ArrayKeys.RIBO_GT: 'volumes/labels/ribosomes',
                    ArrayKeys.RIBO_MASK_UP: 'volumes/ribosomes_mask'
                },
                array_specs={
                    ArrayKeys.MASK_UP: ArraySpec(interpolatable=False),
                    ArrayKeys.RIBO_MASK_UP: ArraySpec(interpolatable=False)
                }
            )


        data_providers.append(n5_source)

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)


    inputs = dict()
    inputs[net_io_names['raw']] = ArrayKeys.RAW
    inputs[net_io_names['mask']] = ArrayKeys.MASK
    inputs[net_io_names['ribo_mask']] = ArrayKeys.RIBO_MASK
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

    request.add(ArrayKeys.RAW, input_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RAW_UP, input_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.GT_LABELS, output_size,  voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK_UP, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.MASK, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RIBO_MASK, output_size, voxel_size=voxel_size_orig)
    request.add(ArrayKeys.RIBO_MASK_UP, output_size, voxel_size=voxel_size_up)
    request.add(ArrayKeys.RIBO_GT, output_size, voxel_size=voxel_size_up)
    for label in labels:
        request.add(label.gt_dist_key, output_size, voxel_size=voxel_size_orig)
        snapshot_request.add(label.pred_dist_key, output_size)
        if label.scale_loss:
            request.add(label.scale_key, output_size, voxel_size=voxel_size_orig)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(ArrayKeys.RAW_UP) + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(ArrayKeys.RAW_UP, context) +
        RandomLocation() + # chose a random location inside the provided arrays
        Reject(ArrayKeys.MASK_UP,min_masked=0.25)
        #Reject(ArrayKeys.MASK) # reject batches wich do contain less than 50% labelled data
        for provider in data_providers)

    train_pipeline = (
        data_sources +
        RandomProvider(labeled_voxels) +
        ElasticAugment((100, 100, 100), (10., 10., 10.), (0, math.pi/2.0),
                       prob_slip=0, prob_shift=0, max_misalign=0,
                       subsample=8) +
        SimpleAugment() +
        #ElasticAugment((40, 1000, 1000), (10., 0., 0.), (0, 0), subsample=8) +
        IntensityAugment(ArrayKeys.RAW_UP, 0.95, 1.05, -0.05, 0.05) +
        IntensityScaleShift(ArrayKeys.RAW_UP, 2, -1) +
        ZeroOutConstSections(ArrayKeys.RAW_UP))

    for label in labels:
        if label.labelname != 'ribosomes':
            train_pipeline += AddDistance(label_array_key=ArrayKeys.GT_LABELS,
                                          distance_array_key=label.gt_dist_key,
                                          normalize='tanh',
                                          normalize_args=dt_scaling_factor,
                                          label_id=label.labelid, factor=2)
        else:
            train_pipeline += AddDistance(label_array_key=ArrayKeys.RIBO_GT,
                                          distance_array_key=label.gt_dist_key,
                                          normalize='tanh+',
                                          normalize_args=(dt_scaling_factor, 8),
                                          label_id=label.labelid, factor=2)

    train_pipeline = (train_pipeline +
                      DownSample(ArrayKeys.MASK_UP, 2, ArrayKeys.MASK)+
                      DownSample(ArrayKeys.RIBO_MASK_UP, 2, ArrayKeys.RIBO_MASK) )
    for label in labels:
        if label.scale_loss:
            if label.labelname != 'ribosomes':
                train_pipeline += BalanceByThreshold(label.gt_dist_key, label.scale_key, mask=ArrayKeys.MASK)
            else:
                train_pipeline += BalanceByThreshold(label.gt_dist_key, label.scale_key, mask=ArrayKeys.RIBO_MASK)

    train_pipeline = (
        train_pipeline +
        DownSample(ArrayKeys.RAW_UP, 2, ArrayKeys.RAW) +
        PreCache(
            cache_size=30,
            num_workers=30) +
        Train(
            'unet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names[loss_name],
            inputs=inputs,
            summary=net_io_names['summary'],
            log_dir='log',
            outputs=outputs,
            gradients={},
            log_every=5,
            save_every=500
        ) +
        Snapshot(
            snapshot,
            every=500,
            output_filename='batch_{iteration}.hdf',
            output_dir='snapshots/',
            additional_request=snapshot_request) +

        PrintProfilingStats(every=500))

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5"
    data_sources = ['hela_cell2_crop1_122018',
                    'hela_cell2_crop3_122018',
                    'hela_cell2_crop6_122018',
                    'hela_cell2_crop7_122018',
                    'hela_cell2_crop8_122018',
                    'hela_cell2_crop9_122018',
                    'hela_cell2_crop13_122018',
                    'hela_cell2_crop14_122018',
                    'hela_cell2_crop15_122018',
                    'hela_cell2_crop18_122018',
                    'hela_cell2_crop19_122018',
                    'hela_cell2_crop20_122018',
                    'hela_cell2_crop21_122018',
                    'hela_cell2_crop22_122018'
                    ]
    labeled_voxels = (500*500*100,
                      400*400*250,
                      250*250*250,
                      300*300*80,
                      200*200*100,
                      100*100*53,
                      160*160*110,
                      150*150*65,
                      150*150*64,
                      200*200*110,
                      150*150*55,
                      200*200*85,
                      160*160*55,
                      170*170*100,
                      )
    ribo_sources = ['hela_cell2_crop6_122018',
                    'hela_cell2_crop7_122018',
                    'hela_cell2_crop13_122018',
                    ]

    input_shape = (196, 196, 196)
    output_shape = (92, 92, 92)
    dt_scaling_factor = 50
    max_iteration = 500000
    loss_name = 'loss_total'

    labels = []
    labels.append(Label('ecs', 1, data_sources=data_sources))
    labels.append(Label('plasma_membrane', 2, data_sources=data_sources))
    labels.append(Label('mito', (3, 4, 5), data_sources=data_sources))
    labels.append(Label('mito_membrane', 3, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources))
    labels.append(Label('mito_DNA', 5, scale_loss=False, scale_key=labels[-2].scale_key, data_sources=data_sources))
    labels.append(Label('vesicle', (8, 9), data_sources=data_sources))
    labels.append(Label('vesicle_membrane', 8, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources))
    labels.append(Label('MVB', (10, 11), data_sources=data_sources))
    labels.append(Label('MVB_membrane', 10, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('lysosome', (12, 13), data_sources=data_sources))
    labels.append(Label('lysosome_membrane', 12, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources))
    labels.append(Label('LD', (14, 15), data_sources=data_sources))
    labels.append(Label('LD_membrane', 14, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('er', (16, 17, 18, 19, 20, 21), data_sources=data_sources))
    labels.append(Label('er_membrane', (16, 18, 20), scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources))
    labels.append(Label('ERES', (18, 19), data_sources=data_sources))
    labels.append(Label('ERES_membrane', 18, scale_loss=False, scale_key=labels[-1].scale_key,
                        data_sources=data_sources))
    labels.append(Label('nucleus', (20, 21, 24, 25), data_sources=data_sources))
    labels.append(Label('NE', (20, 21), scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('NE_membrane', 20, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    labels.append(Label('chromatin', 24, scale_loss=False, scale_key=labels[-1].scale_key, data_sources=data_sources))
    #labels.append(Label('nucleoplasm', 25, scale_loss=False, scale_key=labels[-1].scale_key,
    #                    data_sources=data_sources)) #stop using this
    labels.append(Label('microtubules', 26, data_sources=data_sources))
    labels.append(Label('ribosomes', 1, data_sources=ribo_sources))

    train_until(max_iteration, data_sources, labeled_voxels, ribo_sources, input_shape, output_shape,
                dt_scaling_factor, loss_name,
                labels)
