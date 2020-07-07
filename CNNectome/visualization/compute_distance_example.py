import zarr
import numpy as np
import logging
import numcodecs
import collections.abc
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from scipy.ndimage import generate_binary_structure


def process(label, label_id, label_voxel_size=(2., 2., 2.), mask_voxel_size=(4., 4., 4.), factor=2, max_distance=2.76*50, scale=50):
    mask = np.ones((np.array(label.shape) * label_voxel_size[0] / mask_voxel_size[0]).astype(np.int))
    binary_label = np.isin(label, label_id)

    dims = binary_label.ndim
    sampling = tuple(float(v) for v in label_voxel_size)
    distances = signed_distance(binary_label, sampling=sampling)

    if isinstance(factor, tuple):
        slices = tuple(slice(None, None, k) for k in factor)
    else:
        slices = tuple(slice(None, None, factor) for _ in range(dims))

    distances = distances[slices]

    distances = clip_distance(distances, (-max_distance, max_distance))

    # modify in-place the label mask
    mask = constrain_distances(mask, distances, mask_voxel_size, max_distance)
    distances = np.tanh(distances / scale)
    return distances, mask


def clip_distance(distances, max_distance):
    if not isinstance(max_distance, tuple):
        max_distance = (-max_distance, max_distance)
    distances = np.clip(distances, max_distance[0], max_distance[1])
    return distances


def signed_distance(label, **kwargs):
    # calculate signed distance transform relative to a binary label. Positive distance inside the object,
    # negative distance outside the object. This function estimates signed distance by taking the difference
    # between the distance transform of the label ("inner distances") and the distance transform of
    # the complement of the label ("outer distances"). To compensate for an edge effect, .5 (half a pixel's
    # distance) is added to the positive distances and subtracted from the negative distances.
    inner_distance = distance_transform_edt(binary_erosion(label, border_value=1,
                                                           structure=generate_binary_structure(label.ndim,
                                                                                               label.ndim)),
                                            **kwargs)
    outer_distance = distance_transform_edt(np.logical_not(label), **kwargs)
    result = inner_distance - outer_distance

    return result


def constrain_distances(mask, distances, mask_sampling, max_distance):
    # remove elements from the mask where the label distances exceed the distance from the boundary
    tmp = np.zeros(np.array(mask.shape) + np.array((2,) * mask.ndim), dtype=mask.dtype)
    slices = tmp.ndim * (slice(1, -1),)
    tmp[slices] = mask
    boundary_distance = distance_transform_edt(binary_erosion(tmp,
                                                              border_value=1,
                                                              structure=generate_binary_structure(tmp.ndim,
                                                                                                  tmp.ndim)),
                                               sampling=mask_sampling)
    boundary_distance = boundary_distance[slices]
    if max_distance is not None:
        boundary_distance = clip_distance(boundary_distance, (-max_distance, max_distance))

    mask_output = mask.copy()
    if max_distance is not None:
        mask_output[(abs(distances) >= boundary_distance) *
                    (distances >= 0) *
                    (boundary_distance < max_distance)] = 0
        mask_output[(abs(distances) >= boundary_distance + 1) *
                    (distances < 0) *
                    (boundary_distance + 1 < max_distance)] = 0
    return mask_output


def extract_whole():
    n5_file = "/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm.n5"
    label_ds = "volumes/groundtruth/0003/Crop3/labels/all"
    n5f = zarr.open(n5_file, mode="r")
    label = np.array(n5f[label_ds])
    label_id = (16, 17, 18, 19, 20, 21, 22, 23)
    distances, mask = process(label, label_id)
    print(np.min(distances))
    print(np.max(distances))

    n5_tgt = "/groups/saalfeld/home/heinrichl/hela2_crop3_distances.n5"
    n5_tgtf = zarr.open(n5_tgt, mode="a")
    n5_tgtf.empty(
        name="er",
        shape=distances.shape,
        compressor=numcodecs.GZip(6),
        dtype=distances.dtype,
        chunks=(64, 64, 64),
    )
    n5_tgtf["er"] = distances
    n5_tgtf.empty(
        name="er_mask",
        shape=mask.shape,
        compressor=numcodecs.GZip(6),
        dtype=mask.dtype,
        chunks=(64, 64, 64),
    )
    n5_tgtf["er_mask"] = mask


def main():
    crop_off = (3999, 200, 7800)
    global_off = (4124, 352, 7980)
    size = (36, 36, 36)
    local_off = tuple(np.array(global_off) - np.array(crop_off))
    sl = tuple(slice(local_off[k], local_off[k]+size[k]) for k in range(3))
    n5 = "/groups/saalfeld/home/heinrichl/hela2_crop3_distances.n5"
    n5_f = zarr.open(n5, mode="r")
    mito_crop = n5_f["mito"][sl]
    mito_mask_crop = n5_f["mito_mask"][sl]
    er_crop = n5_f["er"][sl]
    er_mask_crop = n5_f["er_mask"][sl]
    n5_tgt = "/groups/saalfeld/home/heinrichl/hela2_crop3_distances_cropped.n5"
    n5_tgtf = zarr.open(n5_tgt, mode="a")
    n5_tgtf.empty(
        name="er",
        shape=er_crop.shape,
        compressor=numcodecs.GZip(6),
        dtype=er_crop.dtype,
        chunks=(18, 18, 18),
    )
    n5_tgtf["er"][:] = er_crop[:]
    n5_tgtf.empty(
        name="er_mask",
        shape=er_mask_crop.shape,
        compressor=numcodecs.GZip(6),
        dtype=er_mask_crop.dtype,
        chunks=(18, 18, 18),
    )
    n5_tgtf["er_mask"][:] = er_mask_crop[:]

    n5_tgtf.empty(
        name="mito",
        shape=mito_crop.shape,
        compressor=numcodecs.GZip(6),
        dtype=mito_crop.dtype,
        chunks=(18, 18, 18),
    )
    n5_tgtf["mito"][:] = mito_crop[:]
    n5_tgtf.empty(
        name="mito_mask",
        shape=mito_mask_crop.shape,
        compressor=numcodecs.GZip(6),
        dtype=mito_mask_crop.dtype,
        chunks=(18, 18, 18),
    )
    n5_tgtf["mito_mask"][:] = mito_mask_crop[:]


if __name__ == "__main__":
    main()