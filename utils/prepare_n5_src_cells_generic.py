from __future__ import print_function
import z5py
import h5py
import numpy as np
import collections
import datetime
import logging


def add_ds(target, name, data, chunks, resolution, offset, **kwargs):
    if name not in target:
        logging.info("Writing dataset {0:} to {1:}".format(name, target.path))
        ds = target.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype, compression='gzip',
                                   type='gzip', level=6)
        target[name][:] = np.array(data)
        target[name].attrs['resolution'] = resolution
        target[name].attrs['offset'] = offset
        for k in kwargs:
            target[name].attrs[k] = kwargs[k]
    else:
        logging.info("Dataset {0:} already exists in {1:}, not overwriting".format(name, target.path))
        ds = target[name]
    return ds


def add_subset_label_ds(target, labels, name, label_ids, chunks, resolution):
    if not isinstance(label_ids, collections.Iterable):
        label_ids = (label_ids, )
    add_ds(target, name, np.logical_or.reduce([labels == lid for lid in label_ids]).astype(labels.dtype),
           chunks, resolution, [0., 0., 0.])


def main(orig, target, mapping, ribos=True, src_label_name='volumes/labels/gt', ribo_orig=None):
    raw = orig['volumes/raw']
    if ribo_orig is None:
        ribo_orig=orig
    labels = orig[src_label_name]
    if ribos:
        ribosomes = ribo_orig['volumes/labels/ribosomes']
    if 'volumes' not in target.keys():
        target.create_group('volumes')
    logging.info("RAW dataset {0:} has resolution {1:} and offset {2:}".format(raw.shape, raw.attrs['resolution'],
                                                                          raw.attrs['offset']))
    logging.info("LABELS dataset {0:} has resolution {1:} and offset {2:}".format(labels.shape, labels.attrs['resolution'],
                                                                          labels.attrs['offset']))
    if ribos:
        logging.info("RIBOSOMES dataset {0:} has resolution {1:} and offset {2:}".format(ribosomes.shape,
                                                                                         ribosomes.attrs[
        'resolution'], ribosomes.attrs['offset']))
    cont = np.unique(labels)
    hist = np.histogram(labels, bins=list(cont)+[cont[-1]+0.1])
    logging.info("LABELS contains ids {0:} in freq {1:}".format(cont, hist[0]))
    if ribos:
        cont_ribo = np.unique(ribosomes)
        hist_ribo = np.histogram(ribosomes, bins=list(cont_ribo)+[cont_ribo[-1]+0.1])
        logging.info("RIBOSOMES contains ids {0:} in freq {1:}".format(cont_ribo, hist_ribo[0]))
    logging.info("Doubling resolution of RAW (using nearest neighbor)")
    raw_up = np.repeat(np.repeat(np.repeat(raw, 2, axis=0), 2, axis=1), 2, axis=2)
    logging.info("saving upscaled RAW to {0:}".format(target.path))
    add_ds(target, 'volumes/orig_raw', raw, raw.chunks, list(raw.attrs['resolution']), list(raw.attrs['offset']))
    add_ds(target, 'volumes/raw', raw_up, raw.chunks, [float(r)/2. for r in raw.attrs['resolution']],
           list(raw.attrs['offset']))
    
    padding_before = (((np.array(labels.attrs['offset']) - np.array(labels.attrs['resolution'])/2.) + np.array(
        raw.attrs['resolution']/2.)) / np.array(labels.attrs['resolution'])).astype(np.int)
    padding_after = (np.array(target['volumes/raw'].shape)-padding_before-np.array(labels.shape)).astype(np.int)
    padding = tuple((b, a) for b, a in zip(padding_before, padding_after))
    bg_label = 18446744073709551613


    logging.info("padding LABELS with {0:} to match shape of upscaled RAW, padding value {1:} and relabeling "
                 "using mapping {2:} to {3:}".format(padding, bg_label, range(len(mapping)), mapping))
    # labels_padded = np.pad(labels, padding, 'constant', constant_values=bg_label)
    # numpy.pad has a bug when handling uint64, it is fixed in the current master so should be good with the next
    # numpy release (currently 1.14.3)

    labels_padded = np.ones((rs*2 for rs in raw.shape), dtype=np.uint64)*bg_label
    labels_padded[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]] = \
        mapping[np.array(labels)]
    cont_relabeled = np.unique(labels_padded)
    hist_relabeled = np.histogram(labels_padded, bins= list(cont_relabeled)+[cont_relabeled[-1]+0.1])

    logging.info("padded LABELS contains ids {0:} in freq {1:}".format(cont_relabeled, hist_relabeled[0]))
    assert raw_up.shape == labels_padded.shape
    if ribos:
        ribosomes_padded = np.ones(labels_padded.shape, dtype=np.uint64)*bg_label
        ribosomes_padded[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]] = \
             np.array(ribosomes)
        ribosomes_mask_padded = np.zeros(labels_padded.shape, dtype=np.uint64)
        ribosomes_mask_padded[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]]\
            = np.ones(ribosomes.shape)
        cont_ribo_relabeled = np.unique(ribosomes_padded)
        hist_ribo_relabeled = np.histogram(ribosomes_padded, bins=list(cont_ribo_relabeled)+[cont_ribo_relabeled[-1]+0.1])
    else:
        ribosomes_mask_padded = np.zeros(labels_padded.shape, dtype=np.uint64)
    if 'labels' not in target['volumes']:
        target['volumes'].create_group('labels')
    add_ds(target, 'volumes/labels/all', labels_padded, labels.chunks, list(labels.attrs['resolution']), [0., 0., 0.],
           orig_ids=list(hist[1]), orig_counts=list(hist[0]), relabeled_ids=list(hist_relabeled[1]),
           relabeled_counts=list(hist_relabeled[0]), mapping=list(mapping))

    add_ds(target, 'volumes/mask', (labels_padded != bg_label).astype(labels.dtype), labels.chunks,
           list(labels.attrs['resolution']), [0., 0., 0.])

    del labels_padded
    if ribos:
         add_ds(target, 'volumes/labels/ribosomes', ribosomes_padded, ribosomes.chunks,
                list(ribosomes.attrs['resolution']), [0., 0., 0.,], orig_ids=list(hist_ribo[1]), orig_counts=list(
                hist_ribo[0]), relabeled_ids = list(hist_ribo_relabeled[1]), relabeled_counts=list(
                hist_relabeled[0]))
    add_ds(target, 'volumes/ribosomes_mask', ribosomes_mask_padded, labels.chunks,
            list(labels.attrs['resolution']), [0., 0., 0.])

    # add_subset_label_ds(target, labels_padded, 'volumes/labels/ECS', (6, 7),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/cell', (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/plasma_membrane', 5,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/ERES', (12, 13),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/ERES_membrane', 12,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/mvb', (3, 9),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/mvb_membrane', 3,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/er', (4, 8),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/er_membrane', 4,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/mito', (1, 2),
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/mito_membrane', 2,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/vesicles', 10,
    #                     labels.chunks, list(labels.attrs['resolution']))
    # add_subset_label_ds(target, labels_padded, 'volumes/labels/microtubules', 11,
    #                     labels.chunks, list(labels.attrs['resolution']))
    orig.close()


def main_cell2_crop1():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop1_Periphery'
                     '/Cell2_Crop1_1012x1012x612+6210-31+344.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop1_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    # mapping = np.array([0, 9, 8, 10, 4, 2, 1, 1, 5, 11, 12, 14, 6, 7, 3, 13])
    mapping = np.array([0, 4, 3, 10, 16, 2, 1, 1, 17, 11, 8, 26, 18, 19, 29, 9])
    ribos = False
    main(orig, target, mapping, ribos)
def main_cell2_crop3():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop3_Mitos'
                     '/Cell2_Crop3_912x912x762+7544-56+3743.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop3.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 16, 10, 2, 3, 8, 26, 17, 4, 11, 9, 18, 29, 1, 5, 19, 12, 13])
    ribos=False
    main(orig, target, mapping, ribos)
def main_cell2_crop6():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop6_Ribosome'
                     '/Cell2_Crop6_762x762x762+2764+144+2743_labels-only.h5', 'r')
    ribo_orig = h5py.File('/nrs/saalfeld/heinrichl/cell/Annotations020719/HeLa_Cell2_Crop6_Ribosome'
                     '/Cell2_Crop6_762x762x762+2764+144+2743_ribosomes.h5', 'r')

    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop6.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 16, 17, 3, 4, 10, 11, 29, 18, 19, 26])
    ribos=True
    main(orig, target, mapping, ribos, ribo_orig=ribo_orig)
def main_cell2_crop7():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop7_PM'
                     '/Cell2_Crop7_812x812x592+8024-16-256.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop7.n5'.format(
    datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array(range(972))
    mapping[:11] = np.array([0, 2, 2, 1, 29, 29, 26, 8, 9, 3, 4])
    mapping[971]=2
    ribos=True
    srcds = 'volumes/labels/merged_ids'
    main(orig, target, mapping, ribos, src_label_name=srcds)
def main_cell2_crop8():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop8_ERES001'
                     '/Cell2_Crop8_712x712x612+3129+24+993.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop8_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    # mapping = np.array([0, 4, 6, 7, 14, 3, 5, 12, 13, 10])
    mapping = np.array([0, 16, 18, 19, 26, 29, 17, 8, 9, 10])
    ribos=False
    main(orig, target, mapping, ribos)
def main_cell2_crop9():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop9_ERES002'
                     '/Cell2_Crop9_612x612x565+2644+164+1353.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop9_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    # mapping = np.array([0, 6, 8, 9, 3, 4, 5, 7, 10, 11, 14, 12, 13])
    mapping = np.array([0, 18, 3, 4, 29, 16, 17, 19, 10, 11, 26, 8, 9])
    ribos=False
    main(orig, target, mapping, ribos)
def main_cell2_crop13():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop13_ERES006'
                     '/Cell2_Crop13_672x672x622+4904+504+3676.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop13_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 16, 18, 19, 10, 11, 29, 26, 20, 21, 24, 25, 17, 8, 9])
    ribos=True
    main(orig, target, mapping, ribos)

def main_cell2_crop14():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop14_ERES007'
                     '/Cell2_Crop14_662x662x577+6074+119+4160.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop14_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    # mapping = np.array([0, 4, 7, 6, 14, 3, 5, 12, 13])
    mapping = np.array([0, 16, 19, 18, 26, 29, 17, 8, 9])
    ribos=False
    main(orig, target, mapping, ribos)
def main_cell2_crop15():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop15_ERES008'
                     '/Cell2_Crop15_662x662x576+5874+189+4154.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop15_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    # mapping = np.array([0, 4, 5, 6, 7, 3, 14])
    mapping = np.array([0, 16, 17, 18, 19, 29, 26])
    ribos=False
    main(orig, target, mapping, ribos)
def main_cell2_crop18():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations//HeLa_Cell2_Crop18_MVB'
                     '/Cell2_Crop18_712x712x622+1654-56+3474.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop18_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 26, 29, 16, 17, 10, 11])
    ribos = False
    main(orig, target, mapping, ribos)


def main_cell2_crop19():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop19_BadLD'
                     '/Cell2_Crop19_662x662x567+6644+144+4679.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop19_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 16, 17, 14, 15, 26, 8, 9, 29])
    ribos = False
    main(orig, target, mapping, ribos)


def main_cell2_crop20():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations//SUM159_Cell2_Crop20_LD001'
                     '/Cell2_Crop20_712x712x597+4449+489+5245.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop20_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 3, 4, 14, 15, 16, 17, 10, 11, 29])
    ribos = False
    main(orig, target, mapping, ribos)


def main_cell2_crop21():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations//SUM159_Cell2_Crop21_LD002'
                     '/Cell2_Crop21_672x672x567+4304+374+5320.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop21_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 3, 4, 16, 17, 14, 15, 29])
    ribos = False
    main(orig, target, mapping, ribos)


def main_cell2_crop22():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/SUM159_Cell2_Crop22_LD003'
                     '/Cell2_Crop22_682x682x612+3754+204+5124.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop22_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 29, 16, 17, 14, 15, 10, 11, 26])
    ribos = False
    main(orig, target, mapping, ribos)
def run_main():
    logging.basicConfig(level=logging.INFO)

    main_cell2_crop1()
    main_cell2_crop3()
    main_cell2_crop6()
    main_cell2_crop7()
    main_cell2_crop8()
    main_cell2_crop9()
    main_cell2_crop13()
    main_cell2_crop14()
    main_cell2_crop15()
    main_cell2_crop18()
    main_cell2_crop19()
    main_cell2_crop20()
    main_cell2_crop21()
    main_cell2_crop22()

if __name__ == '__main__':
    run_main()