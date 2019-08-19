import h5py
import z5py
import numpy as np
import os
import sys

# import time


def match_slice_by_slice(large_ds, patterns_ds, axis_orders):
    patterns = [np.array(pattern_ds) for pattern_ds in patterns_ds]
    values = [pattern[0, 0, 0] for pattern in patterns]
    print("looking for values", values)
    # plane_of_ds = np.zeros((1, large_ds.shape[1], large_ds.shape[2]), dtype=np.uint8)
    finds = [False] * len(patterns_ds)
    no_match_found = len(patterns_ds)
    print(large_ds.shape)
    for z_in_ds in range(0, large_ds.shape[0], large_ds.chunks[0]):
        tile_slice = (
            slice(z_in_ds, z_in_ds + large_ds.chunks[0], None),
            slice(0, large_ds.shape[1], None),
            slice(0, large_ds.shape[2], None),
        )
        tile = np.array(large_ds[tile_slice])
        print("Tile loaded")
        for z_in_tile in range(0, large_ds.chunks[0]):
            z = z_in_ds + z_in_tile
            if z % 10 == 0:
                print(z, end="\n")
                sys.stdout.flush()
            print(".", end="")
            if no_match_found > 0:
                # for y in range(large_ds.shape[1]):
                plane_slice = (
                    slice(z_in_tile, z_in_tile + 1, None),
                    slice(0, large_ds.shape[1], None),
                    slice(0, large_ds.shape[2], None),
                )
                plane_of_ds = tile[plane_slice]
                # large_ds.read_direct(plane_of_ds, plane_slice, slice(None, None, None))
                # plane_of_ds = plane_of_ds.squeeze()
                # print(plane_of_ds.shape)
                assert plane_of_ds.shape[0] == 1
                for y in range(plane_of_ds.shape[0]):
                    for x in range(plane_of_ds.shape[1]):
                        v = plane_of_ds[0, y, x]
                        for pattern_num, v_target in enumerate(values):
                            if v_target is not None and v == v_target:
                                # print("checking at", z, y, x)
                                found, aos = match_one_block_arr(
                                    plane_of_ds,
                                    patterns[pattern_num][:5, :5, :5],
                                    (0, y, x),
                                    axis_orders=axis_orders,
                                )
                                if found:
                                    print("good candidate at", z, y, x)
                                    found, ao = match_one_block_ds(
                                        large_ds,
                                        patterns[pattern_num],
                                        (z, y, x),
                                        axis_orders=aos,
                                    )
                                    if found:
                                        finds[pattern_num] = tuple(
                                            (z, y, x)[a] for a in ao
                                        )
                                        values[pattern_num] = None
                                        print("FOUND ONE", finds)
                                        no_match_found -= 1
                                        break
                                    else:
                                        print("...but nope")
            else:
                break
    return finds


# def match_blocks(large_ds, pattern_ds, candidate_locations, axis_orders=[(0,1,2)]):
#    if len(candidate_locations)>0:
#        pattern_arr = np.array(pattern_ds)
#        pattern_shape_zyx = pattern_arr.shape


def match_one_block_ds(
    large_ds, pattern_arr, candidate, axis_orders=(0, 1, 2), verbose=True
):
    pattern_shape_zyx = pattern_arr.shape
    for ao in axis_orders:
        pattern_shape = tuple(pattern_shape_zyx[ao_zyx] for ao_zyx in ao)

        # print(candidate)
        if np.array(
            [
                candidate[k] + pattern_shape[k] <= large_ds.shape[k]
                for k in range(len(candidate))
            ]
        ).all():
            candidate_block = np.zeros(pattern_shape, dtype=np.uint8)
            slice_from_large = (
                slice(candidate[0], candidate[0] + pattern_shape[0], None),
                slice(candidate[1], candidate[1] + pattern_shape[1], None),
                slice(candidate[2], candidate[2] + pattern_shape[2], None),
            )
            slice_to_dest = (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
            )
            # print(candidate_block.shape, candidate_block.shape)

            candidate_block[slice_to_dest] = large_ds[slice_from_large]
            if (candidate_block == pattern_arr.transpose(ao)).all():
                if verbose:
                    print()
                    print(
                        tuple(
                            str(candidate[k]) + "/" + str(large_ds.shape[k])
                            for k, ao_zyx in enumerate(ao)
                        )
                    )
                    print(tuple(pattern_shape[ao_zyx] for ao_zyx in ao))
                    print(ao)
                return True, ao
    return False, None


def match_one_block_arr(arr_sl, pattern, candidate, axis_orders=(0, 1, 2)):
    match_aos = []
    found = False
    for ao in axis_orders:
        pattern_shape = pattern.transpose(ao).shape
        # print(candidate, pattern_shape, arr_sl.shape)
        if np.array(
            [candidate[k] + pattern_shape[k] <= arr_sl.shape[k] for k in [1, 2]]
        ).all():

            slice_from_large = (
                slice(0, 1, None),
                slice(candidate[1], candidate[1] + pattern_shape[1], None),
                slice(candidate[2], candidate[2] + pattern_shape[2], None),
            )
            pattern_sl = (
                slice(0, 1, None),
                slice(None, None, None),
                slice(None, None, None),
            )

            if (arr_sl[slice_from_large] == pattern.transpose(ao)[pattern_sl]).all():
                found = True
                match_aos.append(ao)
    return found, match_aos


def test():
    # generate random uint8 data
    print("Preparing test...")
    data = np.random.random_integers(0, 255, (1000, 1001, 1002)).astype(np.uint8)

    # generate two crops
    crop_loc1 = (314, 262, 537)
    crop_size1 = (118, 128, 226)
    crop_slice1 = (
        slice(crop_loc1[0], crop_loc1[0] + crop_size1[0], None),
        slice(crop_loc1[1], crop_loc1[1] + crop_size1[1], None),
        slice(crop_loc1[2], crop_loc1[2] + crop_size1[2], None),
    )
    crop_loc2 = (371, 523, 167)
    crop_size2 = (352, 221, 348)
    crop_slice2 = (
        slice(crop_loc2[0], crop_loc2[0] + crop_size2[0], None),
        slice(crop_loc2[1], crop_loc2[1] + crop_size2[1], None),
        slice(crop_loc2[2], crop_loc2[2] + crop_size2[2], None),
    )
    crop_loc3 = (319, 861, 225)
    crop_size3 = (435, 37, 170)
    crop_slice3 = (
        slice(crop_loc3[0], crop_loc3[0] + crop_size3[0], None),
        slice(crop_loc3[1], crop_loc3[1] + crop_size3[1], None),
        slice(crop_loc3[2], crop_loc3[2] + crop_size3[2], None),
    )

    data_crop1 = data[crop_slice1]
    data_crop2 = data[crop_slice2]
    data_crop3 = data[crop_slice3].transpose(1, 0, 2)
    crop_loc3 = tuple(crop_loc3[a] for a in (1, 0, 2))
    crop_size3 = tuple(crop_size3[a] for a in (1, 0, 2))

    assert data_crop1.shape == crop_size1
    assert data_crop2.shape == crop_size2
    assert data_crop3.shape == crop_size3

    ## saving
    with h5py.File(os.path.expanduser("~/tmp/test_largeds.h5"), "w") as f:
        f.create_dataset("main", data=data)
    with h5py.File(os.path.expanduser("~/tmp/test_crop1.h5"), "w") as f:
        f.create_dataset("main", data=data_crop1)
    with h5py.File(os.path.expanduser("~/tmp/test_crop2.h5"), "w") as f:
        f.create_dataset("main", data=data_crop2)
    with h5py.File(os.path.expanduser("~/tmp/test_crop3.h5"), "w") as f:
        f.create_dataset("main", data=data_crop3)

    print("...succesfully")
    print("Running test")
    large_ds_file = (os.path.expanduser("~/tmp/test_largeds.h5"), "main")
    pattern_files = [
        (os.path.expanduser("~/tmp/test_crop1.h5"), "main"),
        (os.path.expanduser("~/tmp/test_crop2.h5"), "main"),
        (os.path.expanduser("~/tmp/test_crop3.h5"), "main"),
    ]
    finds = main(
        large_ds_file,
        pattern_files,
        axis_orders=[(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    )
    try:
        assert finds[0] == crop_loc1
        assert finds[1] == crop_loc2
        assert finds[2] == crop_loc3
        print("Passed")
    except AssertionError:
        print("Failed")
    print("Cleaning up after myself...")
    os.remove(large_ds_file[0])
    for f in pattern_files:
        os.remove(f[0])
    print("...done")


def main(large_h5, patterns_h5, axis_orders=[(0, 1, 2)]):
    # test()

    # read dense gt
    lds_f = z5py.File(large_h5[0], use_zarr_format=False)
    large_ds = lds_f[large_h5[1]]

    patterns_ds = []
    pds_fs = []
    for pattern_h5 in patterns_h5:
        pds_fs.append(h5py.File(pattern_h5[0], "r"))
        patterns_ds.append(pds_fs[-1][pattern_h5[1]])

    finds = match_slice_by_slice(large_ds, patterns_ds, axis_orders=axis_orders)

    # read large ds
    # get values
    # lds_f.close()
    for pds_f in pds_fs:
        pds_f.close()
    return finds
    # idx_dict = match_slice_by_slice(large_ds, values)

    # for pattern_num, candidate_locations in idx_dict.iteritems():
    #    match_blocks(patterns[pattern_num], candidate_locations)


if __name__ == "__main__":
    # large_ds_file = ('/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale.h5', 'data')
    # pattern_files = [('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/trvol-250-1-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/trvol-250-2-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/im_uint8.h5', 'main'),
    #                 ('/groups/turaga/turagalab/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/im_uint8.h5', 'main')]
    large_ds_file = ("/nrs/turaga/funkej/fib19/fib19.n5", "volumes/raw/s0")
    pattern_files = [
        ("/nrs/turaga/funkej/fib19/dense_labelled/cube01.hdf", "volumes/raw"),
        ("/nrs/turaga/funkej/fib19/dense_labelled/cube02.hdf", "volumes/raw"),
        ("/nrs/turaga/funkej/fib19/dense_labelled/cube03.hdf", "volumes/raw"),
    ]
    finds = main(
        large_ds_file,
        pattern_files,
        axis_orders=[(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
    )
    print(finds)
    # test()
    # main(('/groups/saalfeld/saalfeldlab/larissa/data/fib25/grayscale.h5', 'data'), '/')
