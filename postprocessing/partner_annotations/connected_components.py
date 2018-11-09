import z5py
import os
import numpy as np
import scipy.ndimage
import logging

def cc(filename_src, dataset_src, filename_tgt, dataset_tgt):

    srcf = z5py.File(filename_src, use_zarr_format=False)
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = z5py.File(filename_tgt, use_zarr_format=False)
    tgtf.create_dataset(dataset_tgt,
                        shape=srcf[dataset_src].shape,
                        compression='gzip',
                        dtype='uint64',
                        chunks=srcf[dataset_src].chunks
                        )
    data = np.array(srcf[dataset_src][:])
    tgt = np.ones(data.shape, dtype=np.uint64)
    maxid = scipy.ndimage.label(data, output=tgt)
    tgtf[dataset_tgt][:] = tgt.astype(np.uint64)
    tgtf[dataset_tgt].attrs['offset'] = srcf[dataset_src].attrs['offset']
    tgtf[dataset_tgt].attrs['max_id'] = maxid


def main():
    thrs_mult = [[153, 76, 76], [127, 63, 63]]
    samples = ['B']#['A+', 'B+', 'C+', 'A', 'B', 'C']
    filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5'
    dataset_srcs = ['predictions_it400000/cleft_dist_cropped_thr{0:}']
    filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5'
    dataset_tgts = ['predictions_it400000/cleft_dist_cropped_thr{0:}_cc']

    for sample in samples:
        logging.info('finding connected components for sample {0:}'.format(sample))
        for thrs in thrs_mult:
            for ds_src, ds_tgt, thr in zip(dataset_srcs, dataset_tgts, thrs):
                logging.info('    dataset {0:}'.format(ds_src.format(thr)))
                cc(filename_src.format(sample), ds_src.format(thr), filename_tgt.format(sample), ds_tgt.format(thr))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
