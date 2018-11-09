from __future__ import print_function
import z5py
import h5py
import numpy as np
import collections


def add_ds(target, name, data, chunks, resolution, offset):
    if name not in target:
        print("Writing dataset {0:} to {1:}".format(name, target.path))
        ds = target.create_dataset(name, shape=data.shape, chunks=chunks, dtype=data.dtype, compression='gzip')
        target[name][:] = data
        target[name].attrs['resolution'] = resolution
        target[name].attrs['offset'] = offset
    else:
        print("Dataset {0:} already exists in {1:}, not overwriting".format(name, target.path))
        ds = target[name]
    return ds

def add_subset_label_ds(target, labels, name, label_ids, chunks, resolution):
    if not isinstance(label_ids, collections.Iterable):
        label_ids = (label_ids, )
    add_ds(target, name, np.logical_or.reduce([labels == lid for lid in label_ids]).astype(labels.dtype),
           chunks, resolution, [0., 0., 0.])


def main():
    orig = h5py.File('/groups/hess/hess_collaborators/Annotations/COS7_Centrosome_8x8x8nm/Centrosome_658x643x588+5646'
                     '+306+3079.aligned.h5', 'r')
    raw = orig['volumes/raw']
    labels = orig['volumes/labels/gt']
    target = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/gt_v1.n5', use_zarr_format=False)
    if 'volumes' not in target.keys():
        target.create_group('volumes')
    print("RAW dataset {0:} has resolution {1:} and offset {2:}".format(raw.shape, raw.attrs['resolution'],
                                                                          raw.attrs['offset']))
    print("LABELS dataset {0:} has resolution {1:} and offset {2:}".format(labels.shape, labels.attrs['resolution'],
                                                                          labels.attrs['offset']))
    cont = np.unique(labels)
    print("LABELS contains ids {0:} in freq {1:}".format(cont, np.histogram(labels, bins=[-1]+list(cont))[0]))
    print("Doubling resolution of RAW (using nearest neighbor)")
    raw_up = np.repeat(np.repeat(np.repeat(raw, 2, axis=0), 2, axis=1), 2, axis=2)
    print("saving upscaled RAW to {0:}".format(target.path))
    add_ds(target, 'volumes/raw', raw_up, raw.chunks, [float(r)/2. for r in raw.attrs['resolution']],
           list(raw.attrs['offset']))

    padding_before = (((np.array(labels.attrs['offset']) - np.array(labels.attrs['resolution'])/2.) + np.array(
        raw.attrs['resolution']/2.)) / np.array(labels.attrs['resolution'])).astype(np.int)
    padding_after = (np.array(target['volumes/raw'].shape)-padding_before-np.array(labels.shape)).astype(np.int)
    padding = tuple((b, a) for b, a in zip(padding_before, padding_after))
    bg_label = 18446744073709551613

    print("padding LABELS with {0:} to match shape of upscaled RAW, padding value {1:}".format(padding, bg_label))
    # labels_padded = np.pad(labels, padding, 'constant', constant_values=bg_label)
    # numpy.pad has a bug when handling uint64, it is fixed in the current master so should be good with the next
    # numpy release (currently 1.14.3)
    labels_padded = np.ones(raw_up.shape, dtype=np.uint64)*bg_label
    labels_padded[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]] = labels
    cont = np.unique(labels_padded)
    print("padded LABELS contains ids {0:} in freq {1:}".format(cont, np.histogram(labels_padded, bins=[-1]+list(cont))[
        0]))
    assert raw_up.shape == labels_padded.shape

    if 'labels' not in target['volumes']:
        target['volumes'].create_group('labels')
    add_ds(target, 'volumes/labels/all', labels_padded, labels.chunks, list(labels.attrs['resolution']), [0., 0., 0.])
    add_ds(target, 'volumes/mask', (labels_padded != bg_label).astype(labels.dtype), labels.chunks,
           list(labels.attrs['resolution']), [0., 0., 0.])
    add_subset_label_ds(target, labels_padded, 'volumes/labels/NA', 0,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/centrosome', 1,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/golgi', (2, 11),
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/golgi_membrane', 11,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/er', (3, 10),
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/er_membrane', 10,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/mvb', (4, 9),
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/mvb_membrane', 9,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/mito', (5, 8),
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/mito_membrane', 8,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/lysosome', (6, 7),
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/lysosome_membrane', 7,
                        labels.chunks, list(labels.attrs['resolution']))
    add_subset_label_ds(target, labels_padded, 'volumes/labels/cytosol', 12,
                        labels.chunks, list(labels.attrs['resolution']))
    orig.close()

if __name__ == '__main__':
    main()