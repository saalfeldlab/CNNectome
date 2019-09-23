from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.contrib import ZeroOutConstSections, AddBoundaryDistance
import tensorflow as tf
import os
import math
import json
import collections


def train_until(
    max_iteration, data_sources, input_shape, output_shape, augmentor, snapshotname
):
    ArrayKey("RAW")
    ArrayKey("GT_LABELS")
    ArrayKey("GT_MASK")
    ArrayKey("TRAINING_MASK")
    ArrayKey("GT_SCALE")
    ArrayKey("LOSS_GRADIENT")
    ArrayKey("GT_DIST")
    ArrayKey("PREDICTED_DIST_LABELS")

    data_providers = []
    cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/"
    for sample in data_sources:

        h5_source = Hdf5Source(
            os.path.join(cremi_dir, "sample_" + sample + "_cleftsorig.hdf"),
            datasets={
                ArrayKeys.RAW: "volumes/raw",
                # ArrayKeys.GT_MASK: 'volumes/masks/groundtruth',
            },
            # array_specs={
            #    ArrayKeys.GT_MASK: ArraySpec(interpolatable=False)
            # }
        )
        data_providers.append(h5_source)

    voxel_size = Coordinate((40, 4, 4))
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    # input_size = Coordinate((132,)*3) * voxel_size
    # output_size = Coordinate((44,)*3) * voxel_size

    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(ArrayKeys.RAW, input_size)
    # request.add(ArrayKeys.GT_MASK, output_size)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider + Normalize(ArrayKeys.RAW) +  # ensures RAW is in float in [0, 1]
        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        # Pad(ArrayKeys.RAW, Coordinate((8, 8, 8)) * voxel_size)+ #Coordinate((100,1800,1700)),
        # Pad(ArrayKeys.GT_MASK, Coordinate((8, 8, 8)) * voxel_size)
        SpecifiedLocation([Coordinate((4000, 7200, 6800))], choose_randomly=False)
        # RandomLocation()
        for provider in data_providers
    )

    # artifact_source = (
    #    Hdf5Source(
    #        os.path.join(cremi_dir, 'sample_ABC_padded_20160501.defects.hdf'),
    #        datasets={
    #            ArrayKeys.RAW:        'defect_sections/raw',
    #            ArrayKeys.ALPHA_MASK: 'defect_sections/mask',
    #        },
    #        array_specs={
    #            ArrayKeys.RAW:        ArraySpec(voxel_size=(40, 4, 4)),
    #            ArrayKeys.ALPHA_MASK: ArraySpec(voxel_size=(40, 4, 4)),
    #        }
    #    ) +
    #    RandomLocation(min_masked=0.05, mask=ArrayKeys.ALPHA_MASK) +
    #    Normalize(ArrayKeys.RAW) +
    #    IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
    #    ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi/2.0), subsample=8) +
    #    SimpleAugment(transpose_only=[1,2])
    # )

    train_pipeline = data_sources + RandomProvider()
    if augmentor is not None:
        if isinstance(augmentor, collections.Iterable):
            print("iterate augmentor")
            for a in augmentor:
                train_pipeline += a
        else:
            train_pipeline += augmentor
    train_pipeline = (
        train_pipeline
        + ElasticAugment(
            (4, 40, 40),
            (0.0, 0.0, 0.0),
            (0, 0),
            prob_slip=0.05,
            prob_shift=0.02,
            max_misalign=20,
            subsample=8,
        )
        + DefectAugment(
            ArrayKeys.RAW,
            prob_missing=0.08,
            prob_low_contrast=0.08,
            prob_artifact=0.08,
            artifact_source=artifact_source,
            artifacts=ArrayKeys.RAW,
            artifacts_mask=ArrayKeys.ALPHA_MASK,
            contrast_scale=0.5,
        )
    )
    # SimpleAugment(transpose_only=[1, 2])+
    # ElasticAugment((4, 40, 40), (0., 2., 2.), (0, math.pi/2.0),
    #               prob_slip=0.05, prob_shift=0.05, max_misalign=10,
    #               subsample=8) +
    # SimpleAugment(transpose_only=[1,2]) +
    # IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
    # DefectAugment(ArrayKeys.RAW,
    #              prob_missing=0.03,
    #              prob_low_contrast=0.01,
    #              prob_artifact=0.03,
    #              artifact_source=artifact_source,
    #              artifacts=ArrayKeys.RAW,
    #              artifacts_mask=ArrayKeys.ALPHA_MASK,
    #              contrast_scale=0.5) +
    # IntensityScaleShift(ArrayKeys.RAW, 2, -1) +
    # ZeroOutConstSections(ArrayKeys.RAW) +
    train_pipeline = (
        train_pipeline
        + Snapshot(
            {ArrayKeys.RAW: "volumes/raw"},
            every=1,
            output_filename="batch_{id}.hdf",
            output_dir=snapshotname + "/",
        )
        + PrintProfilingStats(every=50)
    )

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(max_iteration):
            b.request_batch(request)

    print("Training finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ArrayKey("RAW")
    ArrayKey("ALPHA_MASK")
    data_sources = ["C"]  # , 'B', 'C']
    input_shape = (43, 430, 430)
    output_shape = (23, 218, 218)
    dt_scaling_factor = 50
    max_iteration = 10
    loss_name = "loss_balanced_syn"
    cremi_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/"
    artifact_source = (
        Hdf5Source(
            os.path.join(cremi_dir, "sample_ABC_padded_20160501.defects.hdf"),
            datasets={
                ArrayKeys.RAW: "defect_sections/raw",
                ArrayKeys.ALPHA_MASK: "defect_sections/mask",
            },
            array_specs={
                ArrayKeys.RAW: ArraySpec(voxel_size=(40, 4, 4)),
                ArrayKeys.ALPHA_MASK: ArraySpec(voxel_size=(40, 4, 4)),
            },
        )
        + RandomLocation(min_masked=0.05, mask=ArrayKeys.ALPHA_MASK)
        + Normalize(ArrayKeys.RAW)
        + IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
        + ElasticAugment((4, 40, 40), (0, 2, 2), (0, math.pi / 2.0), subsample=8)
        + SimpleAugment(transpose_only=[1, 2])
    )
    augmentors = [
        None,
        # SimpleAugment(transpose_only=[1, 2]),
        # ElasticAugment((4, 40, 40), (0., 0., 0.), (math.pi/8, math.pi / 2),
        #               prob_slip=0, prob_shift=0, max_misalign=0,
        #               subsample=8),
        # IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True),
        # ElasticAugment((4, 50, 50), (0., 6., 6.), (0, 0),
        #              prob_slip=0, prob_shift=0, max_misalign=0,
        #              subsample=1),
        # (ElasticAugment((4,40,40),(0.,0.,0.), (0,0),
        #                prob_slip = 0.05, prob_shift = 0.02, max_misalign=20,
        #                subsample=8)+
        # DefectAugment(ArrayKeys.RAW,
        #               prob_missing=0.03,
        #               prob_low_contrast=0.01,
        #               prob_artifact=0.03,
        #               artifact_source=artifact_source,
        #               artifacts=ArrayKeys.RAW,
        #               artifacts_mask=ArrayKeys.ALPHA_MASK,
        #               contrast_scale=0.5)),
        # (SimpleAugment(transpose_only=[1, 2]) +
        # IntensityAugment(ArrayKeys.RAW, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        # DefectAugment(ArrayKeys.RAW,
        #              prob_missing=0.03,
        #              prob_low_contrast=0.01,
        #              prob_artifact=0.03,
        #              artifact_source=artifact_source,
        #              artifacts=ArrayKeys.RAW,
        #              artifacts_mask=ArrayKeys.ALPHA_MASK,
        #              contrast_scale=0.5))
    ]
    # names = ['none', 'flip', 'rotate', 'intensity', 'elastic', 'defect']
    names = ["defect"]

    for augmentor, snapshotname in zip(augmentors, names):
        train_until(
            max_iteration,
            data_sources,
            input_shape,
            output_shape,
            augmentor,
            snapshotname,
        )
