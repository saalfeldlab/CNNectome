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


def main(orig, target, mapping):
    raw = orig['volumes/raw']
    labels = orig['volumes/labels/gt']
    if 'volumes' not in target.keys():
        target.create_group('volumes')
    logging.info("RAW dataset {0:} has resolution {1:} and offset {2:}".format(raw.shape, raw.attrs['resolution'],
                                                                          raw.attrs['offset']))
    logging.info("LABELS dataset {0:} has resolution {1:} and offset {2:}".format(labels.shape, labels.attrs['resolution'],
                                                                          labels.attrs['offset']))
    cont = np.unique(labels)
    hist = np.histogram(labels, bins=list(cont)+[cont[-1]+0.1])
    logging.info("LABELS contains ids {0:} in freq {1:}".format(cont, hist[0]))
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
    labels_padded = np.ones(raw_up.shape, dtype=np.uint64)*bg_label
    labels_padded[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]] = \
        mapping[np.array(labels)]
    cont_relabeled = np.unique(labels_padded)
    hist_relabeled = np.histogram(labels_padded, bins= list(cont_relabeled)+[cont_relabeled[-1]+0.1])
    logging.info("padded LABELS contains ids {0:} in freq {1:}".format(cont_relabeled, hist_relabeled[0]))
    assert raw_up.shape == labels_padded.shape

    if 'labels' not in target['volumes']:
        target['volumes'].create_group('labels')
    add_ds(target, 'volumes/labels/all', labels_padded, labels.chunks, list(labels.attrs['resolution']), [0., 0., 0.],
           orig_ids=list(hist[1]), orig_counts=list(hist[0]), relabeled_ids=list(hist_relabeled[1]),
           relabeld_counts=list(hist_relabeled[0]), mapping=list(mapping))

    add_ds(target, 'volumes/mask', (labels_padded != bg_label).astype(labels.dtype), labels.chunks,
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
    mapping = np.array([0, 9, 8, 10, 4, 2, 1, 1, 5, 11, 12, 14, 6, 7, 3, 13])
    main(orig, target, mapping)


def main_cell2_crop8():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop8_ERES001'
                     '/Cell2_Crop8_712x712x612+3129+24+993.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop8_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 4, 6, 7, 14, 3, 5, 12, 13, 10])
    main(orig, target, mapping)


def main_cell2_crop9():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop9_ERES002'
                     '/Cell2_Crop9_612x612x565+2644+164+1353.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop9_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 6, 8, 9, 3, 4, 5, 7, 10, 11, 14, 12, 13])
    main(orig, target, mapping)


def main_cell2_crop14():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop14_ERES007'
                     '/Cell2_Crop14_662x662x577+6074+119+4160.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop14_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 4, 7, 6, 14, 3, 5, 12, 13])
    main(orig, target, mapping)


def main_cell2_crop15():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/BigCat Annotations/HeLa_Cell2_Crop15_ERES008'
                     '/Cell2_Crop15_662x662x576+5874+189+4154.h5', 'r')
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/hela_cell2_crop15_{0:}.n5'.format(
        datetime.date.today().strftime('%m%d%y')), use_zarr_format=False)
    mapping = np.array([0, 4, 5, 6, 7, 3, 14])
    main(orig, target, mapping)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #main_cell2_crop1()
    main_cell2_crop8()
    #main_cell2_crop9()
    #main_cell2_crop14()
    #main_cell2_crop15()