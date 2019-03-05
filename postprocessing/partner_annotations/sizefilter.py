import z5py
import os
import numpy as np
BG_VAL=0
def sizefilter(filename_src, dataset_src, filename_tgt, dataset_tgt, thr, dat_file=None):

    srcf = z5py.File(filename_src, use_zarr_format=False)
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = z5py.File(filename_tgt, use_zarr_format=False)
    grps = ''
    for grp in dataset_tgt.split('/')[:-1]:
        grps += grp
        if not os.path.exists(os.path.join(filename_tgt, grps)):
            tgtf.create_group(grps)
        grps += '/'
    tgtf.create_dataset(dataset_tgt,
                        shape=srcf[dataset_src].shape,
                        compression='gzip',
                        dtype='uint64',
                        chunks=srcf[dataset_src].chunks
                        )
    tgt = np.array(srcf[dataset_src][:])
    ids, counts = np.unique(tgt, return_counts=True)
    if dat_file is not None:
        np.savetxt(dat_file, counts, '%.4g')
    remove_ids = []
    for id, count in zip(ids, counts):
        if count <= thr:
            remove_ids.append(id)

    tgt[np.isin(tgt, remove_ids)] = BG_VAL
    tgtf[dataset_tgt][:] = tgt.astype(np.uint64)
    tgtf[dataset_tgt].attrs['offset'] = srcf[dataset_src].attrs['offset']


def main():
    thrs = [127, 63, 63]
    samples = ['A+', 'B+', 'C+']#['A', 'B', 'C', 'A+', 'B+', 'C+']
    #samples = ['B+', 'C+']
    sf = 750
    offsets = {
         'A+': (37*40, 1176*4, 955*4),
         'B+': (37*40, 1076*4, 1284*4),
         'C+': (37*40, 1002*4, 1165*4),
        'A': (38*40, 942*4, 951*4),
        'B': (37*40, 1165*4, 1446*4),
        'C': (37*40, 1032*4, 1045*4)
    }

    #segf_name = {'A+': 'sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81',
    #             'B+': 'sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87',
    #             'C+': 'sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75',
    #            }

    # filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/{0:}.n5'
    # dataset_src = 'main'
    # filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/{0:}_sizefiltered750.n5'
    # dataset_tgt = 'main'
    # dat_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_sizefilter750.dat'

    #filename_src = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample{0}.n5'
    #dataset_src = 'segmentations/multicut'
    #filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0}.n5'
    #dataset_src = 'volumes/labels/neuron_ids_constis_slf1'
    #filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0}.n5'
    #dataset_tgt = 'volumes/labels/neuron_ids_constis_slf1_sf{0:}'
    #dat_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}_sizefilter{' \
    #           '1:}.dat'

    filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremieval/data2016-aligned/{0:}.n5'
    dataset_src = 'volumes/labels/neuron_ids_constis_slf1_cropped'
    dataset_tgt = 'volumes/labels/neuron_ids_constis_slf1_sf{0:}_cropped'
    for sample in samples:
        print(sample)
        sizefilter(filename_src.format(sample), dataset_src, filename_src.format(sample),
                   dataset_tgt.format(sf), sf )#dat_file.format(sample, sf))

if __name__ == '__main__':
    main()
