import numpy as np
import typing
import skimage.metrics
import typing
import logging


def crop_to(arr: np.ndarray, target_shape: typing.Tuple[int]) -> np.ndarray:
    """
    Center-crops an array to a desired shape. If the difference in shapes is not even the offset of the resulting array
    will be rounded down, i.e. the removed area "in front" will be smaller than the area "behind" the crop.

    Args:
        arr: numpy array to be cropped
        target_shape: desired shape

    Returns:
        Numpy array center-cropped to target_shape
    """
    if arr.shape != target_shape:
        logging.info(
            "Cropping array of shape {0:} to shape {1:}".format(arr.shape, target_shape)
        )
        offset = (np.array(arr.shape) - np.array(target_shape)) // 2
        sl = tuple(slice(o, o + s, 1) for o, s in zip(offset, target_shape))
        return arr[sl]
    else:
        logging.info("Array already has desired shape {0:}".format(target_shape))
        return arr


def compute_metric(arr: np.ndarray, ref_arr: np.ndarray, metric: str) -> float:
    """
    Computes various metric between two arrays.
    Args:
        arr: numpy array on which metric is to be computed
        ref_arr: reference array - may be smaller than `arr`, then `arr` will be center-cropped to the shape of
        `ref_arr`.
        metric: the metric that should be computed - current options are

    Returns:
        The resulting value for the chosen metric.

    """
    logging.info("Computing metric {0:}".format(metric))
    logging.info(
        "Test array shape: {0:}, Reference array shape {1:}".format(
            arr.shape, ref_arr.shape
        )
    )
    arr = crop_to(arr, ref_arr.shape)
    if metric == "structural_similarity":
        return skimage.metrics.structural_similarity(arr, ref_arr)
    else:
        raise NotImplementedError("No implementation for metric {0:}".format(metric))
