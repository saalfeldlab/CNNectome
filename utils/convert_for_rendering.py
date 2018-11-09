from __future__ import print_function
import z5py
import h5py
import numpy as np
import collections

TRANSPARENT= 18446744073709551615
def add_ds_combined(target, name, data, all_data, labelid, chunks, resolution, offset, thr=127):
    if name not in target:
        print("Writing dataset {0:} to {1:}".format(name, target.path))
        ds = target.create_dataset(name, shape=data.shape, chunks=chunks, dtype='uint64', compression='gzip')
        data = np.array(data[:])
        data = (data>thr).astype(np.bool)
        all_data[data] = labelid
        data[data==False] = TRANSPARENT
        target[name][:] = data.astype(np.uint64)
        target[name].attrs['resolution'] = resolution
        target[name].attrs['offset'] = offset

    else:
        print("Dataset {0:} already exists in {1:}, not overwriting".format(name, target.path))
        ds = target[name]
        data = np.array(data[:])
        data = (data>thr).astype(np.bool)
        all_data[data] = labelid
    return ds, all_data

def add_ds_rest(target, name, data, chunks, resolution, offset, thr=127):
    if name not in target:
        print("Writing dataset {0:} to {1:}".format(name, target.path))
        ds = target.create_dataset(name, shape=data.shape, chunks=chunks, dtype='uint64', compression='gzip')
        data = np.array(data[:])
        data = (data>thr).astype(np.uint64)
        data[data == 0] = TRANSPARENT
        target[name][:] = data
        target[name].attrs['resolution'] = resolution
        target[name].attrs['offset'] = offset

    else:
        print("Dataset {0:} already exists in {1:}, not overwriting".format(name, target.path))
        ds = target[name]
    return ds
def add_ds_asis(target, name, data, chunks, resolution, offset):
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

#def add_subset_label_ds(target, labels, name, label_ids, chunks, resolution):
#    if not isinstance(label_ids, collections.Iterable):
#        label_ids = (label_ids, )
#    add_ds(target, name, np.logical_or.reduce([labels == lid for lid in label_ids]).astype(labels.dtype),
#           chunks, resolution, [0., 0., 0.])


def main():
    orig = z5py.File('/nrs/saalfeld/heinrichl/cell/gt_v2.1/0821_01/test_cell2_v1_pred_200000.n5',
                     use_zarr_format=False)
    labels_combined = [('plasma_membrane', 5, 128), ('ERES_membrane', 12, 128), ('MVB_membrane', 3, 128),
                       ('er_membrane',
                                                                                                      4, 128),
               ('mito_membrane',
                                                                                                    2,128), ('vesicles',
                                                                                                         10, 128),
              ('microtubules', 11, 128)]
    labels_combined_all = [('ECS', 6, 128), ('cell', 14, 128), ('er', 8, 128), ('ERES', 13, 128), ('mito', 1, 128),
                           ('MVB', 9, 128),
                           ('plasma_membrane', 5, 128), ('er_membrane',4,  128), ('ERES_membrane', 12, 128),
                           ('mito_membrane',
                                                                                                     2, 128),
                           ('MVB_membrane', 3, 128), ('vesicles', 10, 128), ('microtubules', 11, 128)]
    # labels_combined = [('plasma_membrane', 5, 123), ('ERES_membrane', 12, 123), ('MVB_membrane', 3, 123),
    #                    ('er_membrane',
    #                                                                                                   4, 123),
    #            ('mito_membrane',
    #                                                                                                 2,123), ('vesicles',
    #                                                                                                      10, 128),
    #           ('microtubules', 11, 128)]
    # labels_combined_all = [('ECS', 6, 133), ('cell', 14, 133), ('er', 8, 133), ('ERES', 13, 133), ('mito', 1, 133),
    #                        ('MVB', 9, 133),
    #                        ('plasma_membrane', 5, 123), ('er_membrane',4,  123), ('ERES_membrane', 12, 123),
    #                        ('mito_membrane',
    #                                                                                                  2, 123),
    #                        ('MVB_membrane', 3, 123), ('vesicles', 10, 128), ('microtubules', 11, 128)]

    shape = orig[labels_combined[0][0]].shape
    res = [4.0, 4.0, 4.0]
    offset = [0.0, 0.0, 0.0]
    # labels = orig['volumes/labels/gt']
    target = z5py.File('/nrs/saalfeld/heinrichl/cell/gt_v2.1/0821_01/test_cell2_v1_pred_200000_render.n5',
                       use_zarr_format=False)
    if 'volumes' not in target.keys():
        target.create_group('volumes')
    if 'labels' not in target['volumes']:
         target['volumes'].create_group('labels')
    mem_data = np.ones(shape, dtype=np.uint64) * TRANSPARENT
    for labelname, labelid, thr in labels_combined:
        _, mem_data = add_ds_combined(target, 'volumes/labels/{0:}'.format(labelname), orig[labelname], mem_data,
                                     labelid, (128, 128, 128), res, offset, thr=thr)

    cont = np.unique(mem_data)
    print("combined ds contains ids ", cont)
    add_ds_asis(target, 'volumes/labels/mem_combined', mem_data, (128,128,128), res, offset)
    del mem_data

    all_data = np.ones(shape, dtype=np.uint64) * TRANSPARENT
    for labelname, labelid, thr in labels_combined_all:
        _, all_data = add_ds_combined(target, 'volumes/labels/{0:}'.format(labelname), orig[labelname], all_data,
                                      labelid, (128, 128, 128), res, offset, thr=thr)

    cont = np.unique(all_data)
    print("combined ds contains ids ", cont)
    add_ds_asis(target, 'volumes/labels/all', all_data, (128, 128, 128), res, offset)
    del all_data

    #for labelname in labels_rest:
    #    add_ds_rest(target, 'volumes/labels/{0:}'.format(labelname), orig[labelname], (128,128,128), res, offset)

    #orig_raw = z5py.File('/groups/saalfeld/saalfeldlab/projects/cell/nrs-data/cell2/test2.n5', use_zarr_format=False)
    orig_raw = z5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cell/test_cell2_v1.n5')
    add_ds_asis(target, 'volumes/raw', orig_raw['volumes/orig_raw'][:], (128,128,128), res, offset)

if __name__ == '__main__':
    main()