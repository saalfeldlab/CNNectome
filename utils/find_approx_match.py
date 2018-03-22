from __future__ import print_function
import h5py
import z5py
import numpy as np
import skimage.feature
import os
import sys
import json
#import time


def match_approx(large_ds, patterns_ds, axis_orders, thr=0.7):

    patterns = [np.array(pattern_ds) for pattern_ds in patterns_ds]
    pattern_patches = []
    results = []
    for p in patterns:
        for ao in axis_orders:
            pattern_patches.append(p[:min(64, p.shape[0]), :min(64, p.shape[1]), :min(64, p.shape[2])].transpose(ao))
            results.append({'locations':[], 'values':[]})
    #plane_of_ds = np.zeros((1, large_ds.shape[1], large_ds.shape[2]), dtype=np.uint8)
    finds = [False]*len(patterns_ds)
    no_match_found = len(patterns_ds)
    print(large_ds.shape)
    for z_in_ds in range(64, large_ds.shape[0] - 64, large_ds.chunks[0]):
        for y_in_ds in range(64, large_ds.shape[1] - 64, large_ds.chunks[1]):
            print("Tiles {0:}, {1:}".format(z_in_ds, y_in_ds), end='')
            for x_in_ds in range(64, large_ds.shape[2], large_ds.chunks[2]):
                tile_slice = (slice(z_in_ds, min(z_in_ds+large_ds.chunks[0], large_ds.shape[0]), None),
                              slice(y_in_ds, min(y_in_ds+large_ds.chunks[1], large_ds.shape[1]), None),
                              slice(x_in_ds, min(x_in_ds+large_ds.chunks[2], large_ds.shape[2]), None))
                tile = np.array(large_ds[tile_slice])
                if tile.std()>0:
                    print("-", end='')
                    sys.stdout.flush()
                    for nump, p in enumerate(pattern_patches):
                        matches = skimage.feature.match_template(tile, p)
                        print(".", end='')
                        loc = np.where(matches > thr)
                        if len(loc[0]) > 0:
                            v = matches[loc]
                            loc_in_ds = (loc[0]+z_in_ds, loc[1]+y_in_ds, loc[2]+x_in_ds)
                            print("\ncheck candidate", nump, "at:", np.where(matches == np.max(v)), "offset", z_in_ds,
                                  y_in_ds, x_in_ds)
                            results[nump]['locations'] = loc_in_ds
                            results[nump]['values'] = v
                            del loc, v, loc_in_ds
                else:
                    print("x", end='')
            print()
    return results


def main(large_h5, patterns_h5, axis_orders=[(0, 1, 2)]):
    #test()

    # read dense gt
    lds_f = z5py.File(large_h5[0], use_zarr_format=False)
    large_ds = lds_f[large_h5[1]]

    patterns_ds = []
    pds_fs = []
    for pattern_h5 in patterns_h5:
        pds_fs.append(h5py.File(pattern_h5[0], 'r'))
        patterns_ds.append(pds_fs[-1][pattern_h5[1]])
    #patterns_ds.append(large_ds[150:200, 300:500, 1800:2000])
    finds = match_approx(large_ds, patterns_ds,
                                 axis_orders=axis_orders)

    # read large ds
    # get values
    #lds_f.close()
    for pds_f in pds_fs:
        pds_f.close()
    return finds
    #idx_dict = match_slice_by_slice(large_ds, values)

    #for pattern_num, candidate_locations in idx_dict.iteritems():
    #    match_blocks(patterns[pattern_num], candidate_locations)


if __name__ == '__main__':
    #large_ds_file = ('/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale.h5', 'data')
    #pattern_files = [('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/trvol-250-1-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/trvol-250-2-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/im_uint8.h5', 'main')]
    large_ds_file = ('/nrs/turaga/funkej/fib19/fib19.n5', 'volumes/raw/s0')
    pattern_files = [('/nrs/turaga/funkej/fib19/dense_labelled/cube01.hdf', 'volumes/raw'),
                     ('/nrs/turaga/funkej/fib19/dense_labelled/cube02.hdf', 'volumes/raw'),
                     ('/nrs/turaga/funkej/fib19/dense_labelled/cube03.hdf', 'volumes/raw')
                     ]
    finds = main(large_ds_file, pattern_files,
                 axis_orders=[(0, 1, 2)])#, (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)])
    print(finds)
    with open("/groups/saalfeld/home/heinrichl/template_matching_fib19.json", 'w') as f:
        json.dump(finds, f)
    #test()
    #main(('/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale.h5', 'data'), '/')