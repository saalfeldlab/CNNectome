import z5py
import os
import logging

#offsets_minicrop = {
#    'A+': (37, 1676, 1598),
#    'B+': (37, 2201, 3294),
#    'C+': (37, 1702, 2135)
#}
#shapes = {
#    'A+': (125, 1529, 1448),
#    'B+': (125, 1701, 2794),
#    'C+': (125, 1424, 1470),
#    'A': (125, 1438, 1322),
#    'B': (125, 1451, 2112),
#    'C': (125, 1578, 1461)
#}
#offsets = {
#     'A+': (37, 1176, 955),
#     'B+': (37, 1076, 1284),
#     'C+': (37, 1002, 1165),
#     'A': (38, 942, 951),
#     'B': (37, 1165, 1446),
#     'C': (37, 1032, 1045)
# }


offsets = dict()
offsets['A'] = {True: (38, 942, 951),   False: (38, 911, 911)}
offsets['B'] = {True: (37, 1165, 1446), False: (37, 911, 911)}
offsets['C'] = {True: (37, 1032, 1045), False: (37, 911, 911)}
offsets['A+'] = {True: (37, 1176, 955), False: (37, 911, 911)}
offsets['B+'] = {True: (37, 1076, 1284), False: (37, 911, 911)}
offsets['C+'] = {True: (37, 1002, 1165), False: (37, 911, 911)}
shapes = dict()
shapes['A'] = {True: (125, 1438, 1322), False: (125, 1250, 1250)}
shapes['B'] = {True: (125, 1451, 2112), False: (125, 1250, 1250)}
shapes['C'] = {True: (125, 1578, 1469), False: (125, 1250, 1250)}
shapes['A+'] = {True: (125, 1556, 1447), False: (125, 1250, 1250)}
shapes['B+'] = {True: (125, 1701, 2792), False: (125, 1250, 1250)}
shapes['C+'] = {True: (125, 1424, 1470), False: (125, 1250, 1250)}



def crop_to_seg(filename_src, dataset_src, filename_tgt, dataset_tgt, offset, shape):
    srcf = z5py.File(filename_src, use_zarr_format=False)
    if not os.path.exists(filename_tgt):
        os.makedirs(filename_tgt)
    tgtf = z5py.File(filename_tgt, use_zarr_format=False)
    grps = ''
    for grp in dataset_tgt.split('/')[:-1]:
        grps +=grp
        if not os.path.exists(os.path.join(filename_tgt,grps)):
            tgtf.create_group(grps)
        grps += '/'
    chunk_size = tuple(min(c, s) for c,s in zip(srcf[dataset_src].chunks, shape))
    if os.path.exists(os.path.join(filename_tgt, dataset_tgt)):
        assert tgtf[dataset_tgt].shape == shape and tgtf[dataset_tgt].dtype == srcf[dataset_src].dtype and tgtf[
        dataset_tgt].chunks == chunk_size
        skip_ds_creation = True

    else:
        skip_ds_creation = False
    if not skip_ds_creation:
        tgtf.create_dataset(dataset_tgt,
                        shape=shape,
                        compression='gzip',
                        dtype=srcf[dataset_src].dtype,
                        chunks=chunk_size
                        )
    bb = tuple(slice(off, off+sh, None) for off, sh in zip(offset, shape))
    tgtf[dataset_tgt][:] = srcf[dataset_src][bb]
    tgtf[dataset_tgt].attrs['offset'] = offset[::-1]


def main():
    samples = ['A', 'B', 'C']
    filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremieval/{0:}/{1:}.n5'
    data_eval = ['data2016-aligned', 'data2016-unaligned', 'data2017-aligned', 'data2017-unaligned']
    dataset_srcs = ['volumes/masks/groundtruth']
    #filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5'
    dataset_tgts = ['volumes/masks/groundtruth_cropped']
    for de in data_eval:
        if 'unaligned' in de:
            aligned = False
        else:
            aligned = True
        for sample in samples:
            logging.info("cropping sample {0:}".format(sample))
            off = offsets[sample][aligned]
            sh = shapes[sample][aligned]
            for ds_src, ds_tgt in zip(dataset_srcs, dataset_tgts):
                logging.info("   dataset {0:}".format(ds_src))
                crop_to_seg(filename_src.format(de, sample), ds_src, filename_src.format(de, sample), ds_tgt, off, sh)


def main_seg():
    samples = ['A', 'B', 'C', 'A+', 'B+', 'C+']#['A', 'C', 'B+', 'C+']
    filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{0:}_padded_20170424.aligned.0bg.n5'
    dataset_src = 'volumes/labels/neuron_ids'
    # filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5'
    # dataset_tgt = 'volumes/labels/neuron_ids_cropped'
    #filename_src = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_warped/sample{0:}.n5'
    #filename_src = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5'
    #dataset_src = 'volumes/labels/neuron_ids_constis_slf1_sf750'
    #dataset_src = 'segmentations/mc_glia_global2'
    #dataset_src = 'segmentations/multicut'
    filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5'
    dataset_tgt = 'volumes/labels/neuron_ids_gt_cropped'
    for sample in samples:
        print(sample)
        off = offsets[sample]
        sh = shapes[sample]
        crop_to_seg(filename_src.format(sample), dataset_src, filename_tgt.format(sample), dataset_tgt, off, sh)
def main_test_blocks():
    samples = ['A+', 'B+', 'C+']
    filename_src = '/groups/saalfeld/saalfeldlab/larissa/data/cremieval/{0:}/{1:}.n5'
    data_eval = ['data2016-aligned', 'data2016-unaligned']
    datasets_srcs = ['volumes/masks/groundtruth', 'segmentation/multicut']
    dataset_tgts = ['volumes/masks/groundtruth_cropped', 'volumes/labels/neuron_ids_constis_cropped']
    for de in data_eval:
        if 'unaligned' in de:
            aligned = False
        else:
            aligned = True
        for sample in samples:
            logging.info("cropping sample {0:}".format(sample))
            off = offsets[sample][aligned]
            sh = shapes[sample][aligned]
            for ds_src, ds_tgt in zip(datasets_srcs, dataset_tgts):
                logging.info("    dataset {0:}".format(ds_src))
                crop_to_seg(filename_src.format(de, sample), ds_src, filename_src.format(de, sample), ds_tgt, off, sh)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
