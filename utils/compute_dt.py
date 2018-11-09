import logging
import numpy as np
import h5py
from scipy.ndimage.morphology import distance_transform_edt

logger = logging.getLogger(__name__)


def find_boundaries(labels):
    # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
    # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
    # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
    # bound.: 00000001000100000001000      2n - 1

    logger.debug("computing boundaries for %s", labels.shape)
    dims = len(labels.shape)
    in_shape = labels.shape
    out_shape = tuple(2*s - 1 for s in in_shape)
    out_slices = tuple(slice(0, s) for s in out_shape)
    boundaries = np.zeros(out_shape, dtype=np.bool)
    logger.info("boundaries shape is %s", boundaries.shape)
    for d in range(dims):
        logger.info("processing dimension %d", d)
        shift_p = [slice(None)]*dims
        shift_p[d] = slice(1, in_shape[d])
        shift_n = [slice(None)]*dims
        shift_n[d] = slice(0, in_shape[d] - 1)
        diff = (labels[shift_p] - labels[shift_n]) != 0
        logger.info("diff shape is %s", diff.shape)
        target = [slice(None, None, 2)]*dims
        target[d] = slice(1, out_shape[d], 2)
        logger.info("target slices are %s", target)
        boundaries[target] = diff
    return boundaries


def normalize(distances, normalize_mode, normalize_args):
   if normalize_mode == 'tanh':
       scale = normalize_args
       return np.tanh(distances/scale)
   else:
       raise NotImplementedError


    #def __normalize(self, gradients, norm):
#
    #    dims = gradients.shape[0]
#
    #    if norm == 'l1':
    #        factors = sum([np.abs(gradients[d]) for d in range(dims)])
    #    elif norm == 'l2':
    #        factors = np.sqrt(
    #                sum([np.square(gradients[d]) for d in range(dims)]))
    #    else:
    #        raise RuntimeError('norm %s not supported'%norm)
#
    #    factors[factors < 1e-5] = 1
    #    gradients /= factors
#
    #def __scale(self, gradients, distances, scale, scale_args):
#
    #    dims = gradients.shape[0]
#
    #    if scale == 'exp':
    #        alpha, beta = self.scale_args
    #        factors = np.exp(-distances*alpha)*beta
#
    #    gradients *= factors


def create_dt(labels, target_file, voxel_size=(1,1,1), normalize_mode=None, normalize_args=None):
    boundaries = 1.0 - find_boundaries(labels)
    print(np.sum(boundaries==0))
    if False:#np.sum(boundaries == 0) == 0:
        max_distance = min(dim * vs for dim, vs in zip(labels.shape, voxel_size))
        if np.sum(labels) == 0:
            distances = - np.ones(labels.shape, dtype=np.float32) * max_distance
        else:
            distances = np.ones(labels.shape, dtype=np.float32) * max_distance

    else:

        # get distances (voxel_size/2 because image is doubled)
        print("compute dt")
        distances = distance_transform_edt(
            boundaries,
            sampling=tuple(float(v) / 2 for v in voxel_size))
        print("type conversion")
        distances = distances.astype(np.float32)

        # restore original shape
        print("downsampling")
        downsample = (slice(None, None, 2),) * len(voxel_size)
        distances = distances[downsample]

        print("signed dt")
        # todo: inverted distance
        distances[labels == 0] = - distances[labels == 0]

    distances = np.expand_dims(distances, 0)

    if normalize_mode is not None:
        print("normalizing")
        distances = normalize(distances, normalize_mode, normalize_args)
    print("saving")
    target_file.create_dataset('data', data=distances.squeeze())
    target_file.close()


if __name__ == '__main__':
    label_data = h5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_C_cleftsonly_bin.hdf',
                           'r')['volumes/labels/clefts']
    print(label_data.shape)
    target_file = h5py.File('/nrs/saalfeld/heinrichl/synapses/tests/gt_xz.h5', 'w')
    if 'resolution' in label_data.attrs:
        voxel_size = tuple(label_data.attrs['resolution'])
        print('yes')
    else:
        voxel_size = (4,4,40)

    normalize_mode = 'tanh'
    scale=50
    print(voxel_size)
    #print(np.sum(np.array(label_data)[:,1500,]))
    create_dt(np.array(label_data)[:,1550-550:1550+550+1,:], target_file, voxel_size,
              normalize_mode=normalize_mode,
              normalize_args=scale)