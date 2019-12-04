import zarr
import numpy as np
import napari
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

def __signed_distance(label, **kwargs):
    inner_distance = distance_transform_edt(binary_erosion(label, border_value=1,
                                                           structure=generate_binary_structure(label.ndim,
                                                                                               label.ndim)),
                                            **kwargs)
    outer_distance = distance_transform_edt(np.logical_not(label), **kwargs)
    result = inner_distance - outer_distance
    return result

def __clip_distance(distances, max_distance):
    if not isinstance(max_distance, tuple):
        max_distance = (-max_distance, max_distance)
    distances = np.clip(distances, max_distance[0], max_distance[1])
    return distances

def __constrain_distances(mask, distances, mask_sampling, max_distance, add_constant):
    tmp = np.zeros(np.array(mask.shape) + np.array((2,)* mask.ndim), dtype=mask.dtype)
    slices = tmp.ndim * (slice(1, -1), )
    tmp[slices] = mask
    boundary_distance = distance_transform_edt(binary_erosion(tmp, border_value = 1,
                                                              structure=generate_binary_structure(tmp.ndim,
                                                                                                  tmp.ndim)),
                                               sampling=mask_sampling)
    boundary_distance = boundary_distance[slices]
    if max_distance is not None:
        if add_constant is None:
            add = 0
        else:
            add = add_constant
        boundary_distance = __clip_distance(boundary_distance, (-max_distance - add, max_distance - add))
    if max_distance is not None:
        mask[(abs(distances) >= boundary_distance) * (distances>=0) * (boundary_distance < max_distance - add)] = 0
        mask[(abs(distances) >= boundary_distance + 1) * (distances<0) * (boundary_distance+1 < max_distance - add)] = 0
    else:
        mask[np.logical_and(abs(distances)>= boundary_distance, distances>=0)] = 0
        mask[np.logical_and(abs(distances) >= boundary_distance + 1, distances<0)] = 0
    return mask


def compute_distance_transform(labels, label_id, add_constant=None, max_distance=None, factor=None, scaling_factor=50):
    voxel_size = (2,2,2)
    sampling = tuple(float(v) for v in voxel_size)
    mask = np.ones((np.array(labels.shape)/2.).astype(np.int))
    if label_id is not None:
        binary_label = np.in1d(labels.ravel(), label_id).reshape(labels.shape)
    else:
        binary_label = labels > 0
    dims = binary_label.ndim

    distances = __signed_distance(binary_label, sampling=sampling)

    if isinstance(factor, tuple):
        slices = tuple(slice(None, None, k) for k in factor)
    else:
        slices = tuple(slice(None, None, factor) for _ in range(dims))

    distances = distances[slices]
    if max_distance is not None:
        if add_constant is None:
            add = 0
        else:
            add = add_constant
        distances = __clip_distance(distances, (-max_distance-add, max_distance-add))

    mask_voxel_size = (4., 4., 4.)
    mask = __constrain_distances(mask, distances, mask_voxel_size, max_distance, add_constant)

    if add_constant is not None:
        distances += add_constant

    distances = np.tanh(distances/scaling_factor)

    return distances, mask

def view(raw, label, distances, masks, labelids):
    with napari.gui_qt():

        raw_sl = raw[3999 + 50, 180:620, 7780:8220]

        border = np.zeros(raw_sl.shape+(4,))
        border[19:-20, 19, 0] = 255
        border[19:-20, 19, 3] = 255
        border[19:-20, -20, 0] = 255
        border[19:-20, -20, 3] = 255
        border[19, 19:-20, 0] = 255
        border[19, 19:-20, 3] = 255
        border[-20, 19:-19, 0] = 255
        border[-20, 19:-19, 3] = 255
        label_sl = label[100, :, :]

        viewer = napari.view_image(raw_sl, name="raw")
        viewer.add_image(border, name="border", rgb=True)
        for k, distance in enumerate(distances):
            viewer.add_image(distance[50, :, :], translate=(20, 20), name="distances_{0:}".format(k))
        for k, mask in enumerate(masks):
            viewer.add_labels(np.logical_not(mask[50, :, :]), name="mask_{0:}".format(k), translate=(20, 20))
        viewer.add_labels(label_sl, name="label", scale=(0.5, 0.5), translate=(20, 20))
        for k, labelid in enumerate(labelids):
            viewer.add_labels(np.in1d(label_sl.ravel(), labelid).reshape(label_sl.shape),
                          name="label_{0:}".format(k), scale=(0.5, 0.5), translate=(20, 20))

def precompute(label, labelids, scaling_factor, distance_dir, mask_dir):
    distance, mask = compute_distance_transform(np.array(label), labelids, max_distance = 2.76* scaling_factor,
                                                factor=2, scaling_factor=scaling_factor)
    distance_store = zarr.N5Store(distance_dir)
    mask_store = zarr.N5Store(mask_dir)
    zarr.array(distance, store=distance_store, chunks=(30,30,30), compression="gzip")
    zarr.array(mask, store=mask_store, chunks=(30,30,30), compression="gzip")
    return distance, mask

def load(label, labelids, scaling_factor, distance_dirs, mask_dirs):
    def load_precomputed(distance_dirs, mask_dirs):
        distances = []
        masks = []
        for distance_dir in distance_dirs:
            distances.append(zarr.open(distance_dir, mode="r"))
        for mask_dir in mask_dirs:
            masks.append(zarr.open(mask_dir, mode="r"))
        return distances, masks
    try:
        return load(distance_dirs, mask_dirs)
    except ValueError:
        distances = []
        masks = []
        for labelid, distance_dir, mask_dir in zip(labelids, distance_dirs, mask_dirs):
            distance, mask = precompute(label, labelid, scaling_factor, distance_dir, mask_dir)
            distances.append(distance)
            masks.append(mask)
        return distances, masks

def main(n5file, label_ds):
    n5f = zarr.open(n5file, mode="r")
    label = n5f[label_ds]
    raw = n5f["volumes/raw"]
    scaling_factor = 50
    labelids = [(16, 17, 18, 19, 20, 21, 22, 23), (3,4,5)]
    distance_dirs = ["/groups/saalfeld/home/heinrichl/tmp/distances.n5",
                     "/groups/saalfeld/home/heinrichl/tmp/distances_mito.n5"]
    mask_dirs = ["/groups/saalfeld/home/heinrichl/tmp/mask.n5",
                 "/groups/saalfeld/home/heinrichl/tmp/mask_mito.n5"]

    distances, masks = load(label, labelids, scaling_factor, distance_dirs, mask_dirs)
    view(raw, label,distances, masks, labelids)
if __name__ == "__main__":
    main("/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm.n5",
         "volumes/groundtruth/0003/Crop3/labels/all")
