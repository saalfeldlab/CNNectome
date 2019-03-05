import z5py
import os
import numpy as np
from find_partners import bbox_ND
import h5py


def find_crop(filename_src, dataset_src, bg_label=0xfffffffffffffffdL):
    if filename_src.endswith('.hdf') or filename_src.endswith('.h5'):
        srcf = h5py.File(filename_src, 'r')
    else:
        srcf = z5py.File(filename_src, use_zarr_format=False)
    bb = bbox_ND(srcf[dataset_src][:]!=bg_label)
    print(srcf[dataset_src].shape)
    off = (bb[0], bb[2], bb[4])
    shape = (bb[1]-bb[0]+1, bb[3]-bb[2]+1, bb[5]-bb[4]+1)
    return off, shape


#def main():
#    thrs = [127, 63, 63]
#    samples = ['A+', 'B+', 'C+']
#    filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_crop.n5'
#    dataset_srcs = ['predictions_it400000/cleft_dist_cropped', 'predictions_it400000/pre_dist_cropped',
#                    'predictions_it400000/post_dist_cropped']
#    filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_crop.n5'
#    dataset_tgts = ['predictions_it400000/cleft_dist_cropped_thr{0:}',
#                    'predictions_it400000/pre_dist_cropped_thr{0:}',
#                    'predictions_it400000/post_dist_cropped_thr{0:}']#

#    for sample in samples:
#        for ds_src, ds_tgt, thr in zip(dataset_srcs, dataset_tgts, thrs):
#            threshold(filename_src.format(sample), ds_src, filename_tgt.format(sample), ds_tgt.format(thr), thr)


def main():
    samples = ['A', 'B', 'C']
    filename_src = '/groups/saalfeld/saalfeldlab/projects/cremi-synaptic-partners/sample_{' \
                   '0:}_padded_20170424.hdf'
    dataset_src = 'volumes/labels/clefts'

    #samples = ['A+', 'B+', 'C+']
    #filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremieval/data2016-aligned/{0:}.n5'
    #dataset_src = 'masks/groundtruth'
    #bg_label = 0

    # filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{' \
    #                '0:}_padded_20170424.aligned.0bg.n5'
    # dataset_src = 'volumes/labels/neuron_ids'
    offsets = dict()
    shapes = dict()
    for sample in samples:
        off, sh = find_crop(filename_src.format(sample), dataset_src)
        offsets[sample] = off
        shapes[sample] = sh
    print(offsets)
    print(shapes)

if __name__ == '__main__':
    main()
