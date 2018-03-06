from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from training.gunpowder_wrappers import prepare_h5source
import malis
import os
import math
import json


#print(tensorflow.__version__)


def train_until(max_iteration, data_sources):
    data_providers = []
    fib25_dir = "/groups/saalfeld/home/funkej/workspace/projects/caffe/run/fib25/01_data/train"
    if 'fib25h5' in data_sources:

        for volume_name in ("tstvol-520-1", "tstvol-520-2", "trvol-250-1", "trvol-250-2"):
            h5_source = Hdf5Source(os.path.join(fib25_dir, volume_name+'.hdf'),
                                   datasets={VolumeTypes.RAW: 'volumes/raw',
                                             VolumeTypes.GT_LABELS:'volumes/labels/neuron_ids',
                                             VolumeTypes.GT_MASK: 'volumes/labels/mask',},
                                   volume_specs={
                                       VolumeTypes.GT_MASK: VolumeSpec(interpolatable=False)
                                   })
            data_providers.append(h5_source)

    fib19_dir = "/groups/saalfeld/saalfeldlab/larissa/fib19"
    if 'fib19h5' in data_sources:
        for volume_name in ("trvol-250", "trvol-600"):
            h5_source = prepare_h5source(fib19_dir, volume_name)
            data_providers.append(h5_source)

    #todo: dvid source

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_MASK')
    register_volume_type('GT_SCALE')
    register_volume_type('GT_AFFINITIES')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')


    voxel_size = Coordinate((8, 8, 8))

    input_size = Coordinate((196,)*3) * voxel_size
    output_size = Coordinate((92,)*3) * voxel_size


    # specifiy which volumes should be requested for each batch
    request = BatchRequest()
    request.add(VolumeTypes.RAW, input_size)
    request.add(VolumeTypes.GT_LABELS, output_size)
    request.add(VolumeTypes.GT_MASK, output_size)
    request.add(VolumeTypes.GT_SCALE, output_size)
    request.add(VolumeTypes.GT_AFFINITIES, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize() + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(
            {
                VolumeTypes.RAW: Coordinate((100, 100, 100)) * voxel_size,
                VolumeTypes.GT_MASK: Coordinate((100, 100, 100)) * voxel_size
            }
        ) +
        RandomLocation() + # chose a random location inside the provided volumes
        Reject() # reject batches wich do contain less than 50% labelled data
        for provider in data_providers
    )

    snapshot_request = BatchRequest({
        VolumeTypes.LOSS_GRADIENT: request[VolumeTypes.GT_LABELS],
        VolumeTypes.PREDICTED_AFFS: request[VolumeTypes.GT_LABELS],
        VolumeTypes.LOSS_GRADIENT: request[VolumeTypes.GT_AFFINITIES]
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
        ElasticAugment([40, 40, 40], [2, 2, 2], [0, math.pi/2.0], prob_slip=0.01, prob_shift=0.05, max_misalign=1,
                       subsample=8) +
        SimpleAugment() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1) +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        GrowBoundary(steps=1) +
        SplitAndRenumberSegmentationLabels() +
        AddGtAffinities(malis.mknhood3d()) +
        BalanceLabels( VolumeTypes.GT_AFFINITIES,
            VolumeTypes.GT_SCALE,
            VolumeTypes.GT_MASK) +
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
            loss=net_io_names['loss'],
            summary=net_io_names['summary'],
            log_dir='./log/',
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs']: VolumeTypes.GT_AFFINITIES,
                net_io_names['loss_weights']: VolumeTypes.GT_SCALE
            },
            outputs={
                net_io_names['affs']: VolumeTypes.PREDICTED_AFFS
            },
            gradients={
                net_io_names['affs']: VolumeTypes.LOSS_GRADIENT
            }) +
        Snapshot({
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_AFFINITIES: 'volumes/labels/affinities',
                VolumeTypes.PREDICTED_AFFS: 'volumes/labels/pred_affinities',
                VolumeTypes.LOSS_GRADIENT: 'volumes/loss_gradient',
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
    train_until(max_iteration, data_sources)