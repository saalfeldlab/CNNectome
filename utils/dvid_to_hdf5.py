import os
from libdvid import DVIDNodeService
import numpy as np
import h5py

server_addres = "slowpoke1:32768"
uuid = "341635bc8c864fa5acbaf4558122c0d5"#"4b178ac089ee443c9f422b02dcd9f2af"

# the dvid server needs to be started before calling this (see readme)
node_service = DVIDNodeService(server_addres, uuid)


def make_rois(start, shape, size):
    n_x = shape[0] // size
    n_y = shape[1] // size
    n_z = shape[2] // size
    roi = []
    # stupid lazy loop....
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                roi.append([
                    [start[0] + x * size, start[1] + y * size, start[2] + z * size],
                    [start[0] + (x + 1) * size, start[1] + (y + 1) * size, start[2] + (z + 1) * size]]
                )
    return roi

def extract_grayscale(dataset_name, global_start, shape, save_folder):

    save_path = os.path.join(save_folder, "%s.h5" % dataset_name)
    rois = make_rois(global_start, shape, 512)
    block_shape = (512, 512, 512)

    with h5py.File(save_path) as f:
        gs = f.create_dataset("data", shape, dtype=np.uint8, chunks=True, compression='gzip')
        with h5py.File(save_path) as f:
            ii=0
            for start, stop in rois:
                print("extracting block %i / %i"%(ii, len(rois)))
                ii+=1
                bb = tuple(slice(start[i], stop[i])for i in range(len(start)))
                print(bb)

                data = node_service.get_gray3D(dataset_name, block_shape, start)
                print(data.shape)
                gs[bb] = data
# extract all labelsd from the rois and store them to h5
def extract_all_labels(dataset_name, global_start, shape, save_folder):

    save_path = os.path.join(save_folder, "%s.h5" % dataset_name)
    rois = make_rois(global_start, shape, 512)
    block_shape = (512, 512, 512)

    with h5py.File(save_path) as f:
        labels = f.create_dataset(
            "data", shape, dtype=np.uint64, chunks=True, compression='gzip'
        )
        ii = 0
        for start, stop in rois:
            print("extracting block %i / %i" % (ii, len(rois)))
            ii += 1
            bb = tuple(slice(start[i], stop[i]) for i in range(len(start)))
            print(bb)
            dvid_data = node_service.get_labels3D(dataset_name, block_shape, start)
            print(type(dvid_data))
            print(dvid_data.shape)
            print(np.unique(dvid_data))
            labels[bb] = dvid_data


if __name__ == '__main__':

    start = np.array([0, 0, 0])
    stop  = np.array([8255,4479,5311])
    shape = stop - start
    save_path = '/groups/saalfeld/saalfeldlab/larissa/data/fib25/'
    shape = tuple(shape)
    start = tuple(start)

    #labels_name = 'google__fib25_groundtruth_roi_eroded50_z5006-8000'
    ds_name='grayscale'
    extract_grayscale(ds_name, start, shape, save_path)
    print(len(make_rois(start, shape, 512)))
