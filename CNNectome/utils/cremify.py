import h5py
import numpy as np
import os

data_dir = "/groups/saalfeld/saalfeldlab/larissa/data/fib25/"
for volume in ["grayscale"]:

    orig_raw = h5py.File(os.path.join(data_dir, volume) + ".h5", "r")["data"]
    # neuron_ids = h5py.File(os.path.join(data_dir, volume , 'groundtruth_seg.h5'), 'r')['stack']
    z_max = orig_raw.shape[0]
    for z in range(0, z_max, 968):
        with h5py.File(volume + "_z_{0:}_{1:}.h5".format(z, z + 1056), "w") as f:
            print("Cremifying " + volume + "...")

            raw = np.array(
                orig_raw[z : min(z + 1056, z_max), :, :]
            )  # , dtype=np.float32)/255.
            # (raw - np.min(raw))/(float(np.max(raw)) - float(np.min(raw)))
            ds = f.create_dataset("/volumes/raw", data=raw, compression="lzf")
            ds.attrs["resolution"] = (8, 8, 8)
            f.close()

        # ds = f.create_dataset(
        #         '/volumes/labels/neuron_ids',
        #         data=neuron_ids,
        #         #compression='lzf'
        # )
        # ds.attrs['resolution'] = (8, 8, 8)
        #
        # ds = f.create_dataset(
        #         '/volumes/labels/mask',
        #         shape=neuron_ids.shape,
        #         #compression='lzf',
        #         dtype=np.float32,
        #         fillvalue=1.,
        # )
        # #ds[:] = 1
        # ds.attrs['resolution'] = (8, 8, 8)
