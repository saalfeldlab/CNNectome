import logging
import argparse
import z5py
import numpy as np
import skimage.transform


def add_ds(target, name, shape, dtype, chunks, resolution, offset, **kwargs):
    logging.info("Preparing dataset {0:} in {1:}".format(name, target.path))
    ds = target.require_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compression='gzip')
    ds.attrs['resolution'] = resolution
    ds.attrs['offset'] = offset
    for k in kwargs:
        ds.attrs[k] = kwargs[k]
    return ds


def generate_integral_mask(filename, mask_ds_name='volumes/masks/training', target_mask_ds_name =
                            'volumes/masks/training_integral'):
    target = z5py.File(filename, use_zarr_format=False)
    mask_ds = target[mask_ds_name]
    integral_mask_ds = add_ds(target, target_mask_ds_name, mask_ds.shape, np.uint64, (8, 8, 8),
                              list(mask_ds.attrs['resolution']), list(mask_ds.attrs['offset']))
    logging.info('Computing integral mask...')
    integral_mask_ds[:] = skimage.transform.integral_image(mask_ds[:])


def main():
    parser = argparse.ArgumentParser(description="Generate an integral mask")
    parser.add_argument('n5file', type=str, help='n5 file that should be processed (full path)')
    parser.add_argument('-i', '--input_mask_ds', type=str, help='specify dataset of mask for computing integral mask',
                        default='volumes/masks/training')
    parser.add_argument('-o', '--output_mask_ds', type=str, help='specify dataset name for integral mask', default='')
    args = parser.parse_args()
    if args.output_mask_ds == '':
        args.output_mask_ds = args.input_mask_ds + '_integral'
    generate_integral_mask(args.n5file, args.input_mask_ds, args.output_mask_ds)


if __name__ == '__main__':
    main()