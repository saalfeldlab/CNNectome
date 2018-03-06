from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from training.gunpowder_wrappers import prepare_h5source
import malis
import os
import math
import json
import tensorflow as tf

def train_until(max_iteration, data_sources):

    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
    else:
        trained_until = 0
    if trained_until >= max_iteration:
        return

    data_providers = []
    fib25_dir = "/groups/saalfeld/home/funkej/workspace/projects/caffe/run/fib25/01_data/train"
    if 'fib25h5' in data_sources:

        for volume_name in ("tstvol-520-1", "tstvol-520-2", "trvol-250-1", "trvol-250-2"):
            h5_source = Hdf5Source(os.path.join(fib25_dir, volume_name + '.hdf'),
                                   datasets={VolumeTypes.RAW: 'volumes/raw',
                                             VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                                             VolumeTypes.GT_MASK: 'volumes/labels/mask', },
                                   volume_specs={
                                       VolumeTypes.GT_MASK: VolumeSpec(interpolatable=False)
                                   })
            data_providers.append(h5_source)

    fib19_dir = "/groups/saalfeld/saalfeldlab/larissa/fib19"
    if 'fib19h5' in data_sources:
        for volume_name in ("trvol-250", "trvol-600"):
            h5_source = prepare_h5source(fib19_dir, volume_name)
            data_providers.append(h5_source)

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    #register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_AFFINITIES')
    #register_volume_type('GT_AFFINITIES_MASK')
    register_volume_type('GT_SCALE')
    register_volume_type('PREDICTED_AFFS_1')
    register_volume_type('PREDICTED_AFFS_2')
    register_volume_type('LOSS_GRADIENT_1')
    register_volume_type('LOSS_GRADIENT_2')

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate((220,)*3) * voxel_size
    output_1_size = Coordinate((132,)*3) * voxel_size
    output_2_size = Coordinate((44,)*3) * voxel_size

    #input_size = Coordinate((66, 228, 228))*(40,4,4)
    #output_1_size = Coordinate((38, 140, 140))*(40,4,4)
    #output_2_size = Coordinate((10, 52, 52))*(40,4,4)

    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size)
    request.add(VolumeTypes.GT_LABELS, output_1_size)
    request.add(VolumeTypes.GT_MASK, output_1_size)
    request.add(VolumeTypes.GT_AFFINITIES, output_1_size)
    #request.add(VolumeTypes.GT_AFFINITIES_MASK, output_1_size)
    request.add(VolumeTypes.GT_SCALE, output_1_size)

    snapshot_request = BatchRequest()
    snapshot_request.add(VolumeTypes.RAW, input_size) # just to center the rest correctly
    snapshot_request.add(VolumeTypes.PREDICTED_AFFS_1, output_1_size)
    snapshot_request.add(VolumeTypes.PREDICTED_AFFS_2, output_2_size)
    snapshot_request.add(VolumeTypes.LOSS_GRADIENT_1, output_1_size)
    snapshot_request.add(VolumeTypes.LOSS_GRADIENT_2, output_2_size)

    data_sources = tuple(
        provider +
        Normalize() +
        Pad(
            {
                VolumeTypes.RAW: Coordinate((100, 100, 100)) * voxel_size,
                VolumeTypes.GT_MASK: Coordinate((100, 100, 100)) * voxel_size
            }
        ) +
        RandomLocation() +
        Reject()
        for provider in data_providers
    )

    train_pipeline = (
        data_sources +
        RandomProvider() +
        ElasticAugment([40, 40, 40], [2, 2, 2], [0, math.pi/2.0], prob_slip=0.01, prob_shift=0.05, max_misalign=1,
                       subsample=8) +
        SimpleAugment() +

        IntensityAugment(0.9, 1.1, -0.1, 0.1) +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections()+
        GrowBoundary(steps=2) +
        SplitAndRenumberSegmentationLabels() +
        AddGtAffinities(
            malis.mknhood3d()) +
        BalanceLabels({
            VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_SCALE
        },
            {
                VolumeTypes.GT_AFFINITIES: VolumeTypes.GT_MASK
            })+
        PreCache(
            cache_size=40,
            num_workers=10) +
        Train(
            'wnet',
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            summary=net_io_names['summary'],
            log_dir='.log',
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs_1']: VolumeTypes.GT_AFFINITIES,
                net_io_names['loss_weights_1']: VolumeTypes.GT_SCALE,
            },
            outputs={
                net_io_names['affs_1']: VolumeTypes.PREDICTED_AFFS_1,
                net_io_names['affs_2']: VolumeTypes.PREDICTED_AFFS_2
            },
            gradients={
                net_io_names['affs_1']: VolumeTypes.LOSS_GRADIENT_1,
                net_io_names['affs_2']: VolumeTypes.LOSS_GRADIENT_2
            }) +
        IntensityScaleShift(0.5, 0.5) +
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PREDICTED_AFFS_1: 'volumes/labels/pred_affinities_1',
                VolumeTypes.PREDICTED_AFFS_2: 'volumes/labels/pred_affinities_2',
                VolumeTypes.LOSS_GRADIENT_1: 'volumes/loss_gradient_1',
                VolumeTypes.LOSS_GRADIENT_2: 'volumes/loss_gradient_2',
            },
            every=500,
            output_filename='batch_{iteration}.hdf',
            additional_request=snapshot_request) +
        PrintProfilingStats(every=1000)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration - trained_until):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    set_verbose(False)
    data_sources = ['fib25h5']
    max_iteration = 400000
    train_until(max_iteration, data_sources)
