import h5py
import sys
import numpy as np
import logging


def pad(labels_in, shape, raw_offset=(0, 0, 0)):
    logging.info('padding...')
    off_vox = (np.array((labels_in.attrs['offset'])-np.array(raw_offset)) /
               np.array(labels_in.attrs['resolution'])).astype(np.int)
    logging.debug('offset in voxel size is {0:}, calculated from offset {1:} and voxel size {2:}'.format(
                    off_vox, labels_in.attrs['offset'], labels_in.attrs['resolution']))
    labels = np.zeros(shape, dtype=np.uint64)
    labels[off_vox[0]: off_vox[0] + labels_in.shape[0],
           off_vox[1]: off_vox[1] + labels_in.shape[1],
           off_vox[2]: off_vox[2] + labels_in.shape[2]] = labels_in[:]
    logging.info('...done')
    return labels


def gt_mask(labels_in, shape, raw_offset=(0, 0, 0)):
    logging.info('generating groundtruth mask...')
    off_vox = (np.array((labels_in.attrs['offset']) - np.array(raw_offset)) /
               np.array(labels_in.attrs['resolution'])).astype(np.int)
    logging.debug('offset in voxel size is {0:}, calculated from offset {1:} and voxel size {2:}'.format(
        off_vox, labels_in.attrs['offset'], labels_in.attrs['resolution']))
    mask = np.zeros(shape, dtype=np.uint64)
    mask[off_vox[0]: off_vox[0] + labels_in.shape[0],
         off_vox[1]: off_vox[1] + labels_in.shape[1],
         off_vox[2]: off_vox[2] + labels_in.shape[2]] = np.ones(labels_in.shape, dtype=np.uint64)
    logging.info('...done')
    return mask


def val_train_mask(labels_in, shape, raw_offset=(0, 0, 0)):
    logging.info('generating training and validation mask...')
    off_vox = (np.array((labels_in.attrs['offset']) - np.array(raw_offset)) /
               np.array(labels_in.attrs['resolution'])).astype(np.int)
    logging.debug('offset in voxel size is {0:}, calculated from offset {1:} and voxel size {2:}'.format(
        off_vox, labels_in.attrs['offset'], labels_in.attrs['resolution']))
    training = np.zeros(shape, dtype=np.uint64)
    validation = np.zeros(shape, dtype=np.uint64)

    training[off_vox[0]: off_vox[0] + labels_in.shape[0],
             off_vox[1]: off_vox[1] + int(round(0.75 * labels_in.shape[1])),
             off_vox[2]: off_vox[2] + labels_in.shape[2]] = np.ones((labels_in.shape[0],
                                                                         int(round(0.75*labels_in.shape[1])),
                                                                         labels_in.shape[2]),
                                                                         dtype=np.uint64)
    validation[off_vox[0]: off_vox[0] + labels_in.shape[0],
               off_vox[1] + int(round(0.75*labels_in.shape[1])): off_vox[1] + labels_in.shape[1],
               off_vox[2]: off_vox[2] + labels_in.shape[2]] = np.ones((labels_in.shape[0],
                                                                       int(round(0.25 * labels_in.shape[1])),
                                                                       labels_in.shape[2]),
                                                                       dtype=np.uint64)
    logging.debug('TRUE voxels in training are {0:}'.format(np.sum(training)))
    logging.debug('TRUE voxels in validation are {0:}'.format(np.sum(validation)))
    logging.info('...done')
    return training, validation


def prepare_out(in_fh, out_fh):

    logging.info('copy raw...')
    out_fh.create_dataset('volumes/raw', data=np.array(in_fh['volumes/raw']), chunks=in_fh['volumes/raw'].chunks)
    for k, v in in_fh['volumes/raw'].attrs.iteritems():
        out_fh['volumes/raw'].attrs.create(k, v)
    logging.info('...done')

    logging.info('copy attributes...')
    for k, v in in_fh.attrs.iteritems():
        out_fh.attrs.create(k, v)
    logging.info('...done')

    logging.info('copy annotations...')

    out_fh.create_group('annotations')
    for k, v in in_fh['annotations'].attrs.iteritems():
        out_fh['annotations'].attrs.create(k, v)

    out_fh.create_group('annotations/comments')
    for k, v in in_fh['annotations/comments'].attrs:
        out_fh['annotations/comments'].attrs.create(k, v)
    out_fh.create_dataset('annotations/comments/comments', data=in_fh['annotations/comments/comments'],
                          chunks=in_fh['annotations/comments/comments'].chunks)
    for k, v in in_fh['annotations/comments/comments'].attrs.iteritems():
        out_fh['annotations/comments/comments'].attrs.create(k, v)
    out_fh.create_dataset('annotations/comments/target_ids', data=in_fh['annotations/comments/target_ids'],
                          chunks=in_fh['annotations/comments/target_ids'].chunks)
    for k, v in in_fh['annotations/comments/target_ids'].attrs.iteritems():
        out_fh['annotations/comments/target_ids'].attrs.create(k, v)

    out_fh.create_group('annotations/presynaptic_site')
    for k, v in in_fh['annotations/presynaptic_site'].attrs:
        out_fh['annotations/presynaptic_site'].attrs.create(k, v)
    out_fh.create_dataset('annotations/presynaptic_site/partners', data=in_fh['annotations/presynaptic_site/partners'])
    for k, v in in_fh['annotations/presynaptic_site/partners'].attrs.iteritems():
        out_fh['annotations/presynaptic_site/partners'].attrs.create(k, v)

    out_fh.create_dataset('annotations/ids', data=in_fh['annotations/ids'], chunks=in_fh['annotations/ids'].chunks)
    for k, v in in_fh['annotations/ids'].attrs.iteritems():
        out_fh['annotations/ids'].attrs.create(k, v)

    out_fh.create_dataset('annotations/locations', data=in_fh['annotations/locations'], chunks=in_fh[
        'annotations/locations'].chunks)
    for k, v in in_fh['annotations/locations'].attrs.iteritems():
        out_fh['annotations/locations'].attrs.create(k, v)

    out_fh.create_dataset('annotations/types', data=in_fh['annotations/types'], chunks=in_fh[
        'annotations/types'].chunks)
    for k, v in in_fh['annotations/types'].attrs.iteritems():
        out_fh['annotations/types'].attrs.create(k, v)

    logging.info('...done')


def convert(in_file, out_file):
    in_fh = h5py.File(in_file, 'r')
    out_fh = h5py.File(out_file, 'w')
    in_labels = in_fh['volumes/labels/neuron_ids']
    in_clefts = in_fh['volumes/labels/clefts']
    in_raw = in_fh['volumes/raw']
    try:
        raw_offset = in_raw.attrs['offset']
    except KeyError:
        raw_offset = (0, 0, 0)
        
    prepare_out(in_fh, out_fh)

    out_labels = out_fh.create_dataset('volumes/labels/neuron_ids', data=pad(in_labels, in_raw.shape, raw_offset),
                                 chunks=in_labels.chunks)
    out_labels.attrs.create('offset', raw_offset)
    out_labels.attrs.create('resolution', in_labels.attrs['resolution'])

    out_clefts = out_fh.create_dataset('volumes/labels/clefts', data=pad(in_clefts, in_raw.shape, raw_offset),
                                 chunks=in_clefts.chunks)
    out_clefts.attrs.create('offset', raw_offset)
    out_clefts.attrs.create('resolution', in_clefts.attrs['resolution'])

    out_gt_mask = out_fh.create_dataset('volumes/masks/groundtruth', data=gt_mask(in_labels, in_raw.shape, raw_offset),
                                        chunks=in_raw.chunks)
    out_gt_mask.attrs.create('offset', raw_offset)
    out_gt_mask.attrs.create('resolution', in_raw.attrs['resolution'])

    train, val = val_train_mask(in_labels, in_raw.shape, raw_offset)
    out_train_mask = out_fh.create_dataset('volumes/masks/training', data=train, chunks=in_raw.chunks)
    out_train_mask.attrs.create('offset', raw_offset)
    out_train_mask.attrs.create('resolution', in_raw.attrs['resolution'])
    out_val_mask = out_fh.create_dataset('volumes/masks/validation', data=val, chunks=in_raw.chunks)
    out_val_mask.attrs.create('offset', raw_offset)
    out_val_mask.attrs.create('resolution', in_raw.attrs['resolution'])
    del train, val


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    convert(file_in, file_out)