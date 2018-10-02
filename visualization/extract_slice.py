import h5py
import matplotlib.image
import numpy as np

def extract_slice(data_file, dataset, slice_, target):
    hf = h5py.File(data_file, 'r')
    print(hf.keys())#[dataset]
    hf_ds = hf[dataset]
    img_slice = hf_ds[:, slice_, :].squeeze()
    print(np.sum(img_slice))
    img_slice = np.clip((img_slice * 128 + 127).round(), 0, 255).astype('uint8')
    img_rgb_ready = np.stack((img_slice, img_slice, img_slice), axis=2)
    print(img_rgb_ready.shape)
    matplotlib.image.imsave(target, img_rgb_ready)


if __name__ == '__main__':
    data_file1 = \
        '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_C_cleftsonly_bin.hdf'
    data_file2 = \
        '/nrs/saalfeld/heinrichl/synapses/cremi_all_0116_01/prediction_cremi_warped_sampleC_550000.hdf'
    data_file_3 = \
        '/groups/saalfeld/saalfeldlab/larissa/data/gunpowder/cremi/gt_xz.h5'
    dataset1 = \
        'volumes/raw'
    dataset2 = \
        'syncleft_dist'
    dataset3 = \
        'data'
    slice_ = 1550
    slice_3 = 550
    target1 = '/groups/saalfeld/home/heinrichl/figures/raw_xz.png'
    target2 = '/groups/saalfeld/home/heinrichl/figures/pred_xz.png'
    target3 = '/groups/saalfeld/home/heinrichl/figures/gt_xz.png'

    #extract_slice(data_file1, dataset1, slice_, target1)
    #extract_slice(data_file2, dataset2, slice_, target2)
    extract_slice(data_file_3, dataset3, slice_3, target3)