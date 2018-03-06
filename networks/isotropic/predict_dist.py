from __future__ import print_function
import sys
import numpy as np
from gunpowder import *
from gunpowder.tensorflow import *
import copy
import os
import json


def predict(iteration, h5_sourcefile, h5_targetfile):
    register_volume_type('RAW')
    register_volume_type('PREDICTED_DIST')

    checkpoint = 'unet_checkpoint_%d'%iteration

    with open('net_io_names.json', 'r') as f:
        net_io_names = json.load(f)

    voxel_size = Coordinate((8, 8, 8))
    input_size = Coordinate((132,)*3) * voxel_size
    output_size = Coordinate((44,)*3) * voxel_size
    context = (input_size - output_size)/2
    roi = Roi(offset=(0, 0, 0), shape=(520, 520, 520))

    source = (Hdf5Source(h5_sourcefile,
                         datasets={VolumeTypes.RAW: 'volumes/raw'},
                         ) +
              Normalize() +
              IntensityScaleShift(2, -1) +
              ZeroOutConstSections()
              )

    with build(source):
        raw_spec = source.spec[VolumeTypes.RAW]
        print(raw_spec.roi)
        print(source)



    # specifiy which volumes should be requested for each batch
    chunk_request = BatchRequest()
    chunk_request.add(VolumeTypes.RAW, input_size)
    chunk_request.add(VolumeTypes.PREDICTED_DIST, output_size)
    chunk_request[VolumeTypes.RAW].voxel_size = voxel_size
    chunk_request[VolumeTypes.PREDICTED_DIST].voxel_size = voxel_size

    pred_dist_vol_spec =VolumeSpec(roi=raw_spec.roi,
                                   voxel_size=raw_spec.voxel_size,
                                   dtype=np.float32)

    predict_pipeline_part2 =(((
            source+
            Predict(
            checkpoint,
            inputs={net_io_names['raw']: VolumeTypes.RAW},
            outputs={
                net_io_names['dist']: VolumeTypes.PREDICTED_DIST
            },
            volume_specs={VolumeTypes.PREDICTED_DIST: pred_dist_vol_spec})) ,(
        copy.deepcopy(source)+
        Fill({VolumeTypes.PREDICTED_DIST: 0.}, volume_specs={VolumeTypes.PREDICTED_DIST: pred_dist_vol_spec})))
        + RejectToFill([VolumeTypes.PREDICTED_DIST],
                                           mask_volume_type=VolumeTypes.RAW, accepted_case_node=Predict,
                       rejected_case_node=Fill))

    predict_pipeline = (
        #(source,
        predict_pipeline_part2 +

        #+
        # Predict(checkpoint,
        #         inputs={net_io_names['raw']: VolumeTypes.RAW},
        #         outputs={
        #                 net_io_names['dist']: VolumeTypes.PREDICTED_DIST
        #             },
        #         volume_specs={VolumeTypes.PREDICTED_DIST: pred_dist_vol_spec})+
        #Fill({VolumeTypes.RAW: 1.},
        #     volume_specs={VolumeTypes.RAW: raw_spec}) +


        #predict_pipeline_part3 +
        Hdf5Write(dataset_names={VolumeTypes.RAW: 'volumes/raw',
                                 VolumeTypes.PREDICTED_DIST: 'volumes/labels/pred_distances'},
                  output_dir=os.path.dirname(h5_targetfile),
                  output_filename=os.path.basename(h5_targetfile),
                  dataset_dtypes={VolumeTypes.RAW: np.float32,
                                  VolumeTypes.PREDICTED_DIST: np.float32}) +
        Scan(chunk_request)
     )

    print("Starting prediction...")
    with build(predict_pipeline) as b:
        #raw_spec = source.spec[VolumeTypes.RAW]
        dist_spec = raw_spec.copy()
        raw_spec.roi = raw_spec.roi.grow(-context, -context)

        # whole_request = BatchRequest({
        #     VolumeTypes.RAW: raw_spec,
        #     VolumeTypes.PREDICTED_DIST: dist_spec
        # })
        # print(whole_request)
        dummy_request = BatchRequest()
        b.request_batch(dummy_request)
    print("Prediction finished")

if __name__ == "__main__":
    set_verbose(False)

    iteration = 316000
    sourcedir = '/groups/saalfeld/saalfeldlab/larissa/data/gunpowder/fib25_test'
    sourcefile = sys.argv[1]
    targetfile = os.path.splitext(sourcefile)[0]+'_prediction_at_%d.hdf' % iteration
    predict(iteration, os.path.join(sourcedir, sourcefile), targetfile)