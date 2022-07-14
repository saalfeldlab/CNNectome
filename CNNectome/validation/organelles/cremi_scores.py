import lazy_property
import numpy as np
import scipy.ndimage

BG = 0
MASK_OUTSIDE = 0


def find_boundaries_masked(arr, mask=None):
    structure = scipy.ndimage.generate_binary_structure(arr.ndim, 1)
    if mask is not None:
        arr[np.logical_not(mask.astype(np.bool))] = 1
    dilated = scipy.ndimage.morphology.binary_dilation(
        arr, structure=structure, mask=mask
    )
    eroded = scipy.ndimage.morphology.binary_erosion(
        arr, structure=structure, mask=mask, border_value=1
    )
    boundaries = dilated != eroded
    boundaries &= arr != BG
    return boundaries


class CremiEvaluator(object):
    def __init__(
        self,
        truth,
        test,
        sampling=(1, 1, 1),
        clip_distance=200,
        tol_distance=40,
        mask=None,
    ):
        self.test = test
        self.truth = truth
        self.sampling = sampling
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance
        self.mask = mask

    @lazy_property.LazyProperty
    def test_mask(self):
        # todo: more involved masking
        test_mask = self.test == BG
        if self.mask is not None:
            test_mask = np.logical_or(test_mask, self.mask == MASK_OUTSIDE)
        return test_mask

    @lazy_property.LazyProperty
    def truth_mask(self):
        truth_mask = self.truth == BG
        if self.mask is not None:
            truth_mask = np.logical_or(truth_mask, self.mask == MASK_OUTSIDE)
        return truth_mask

    @lazy_property.LazyProperty
    def truth_binary(self):
        truth_binary = self.truth != BG
        if self.mask is not None:
            truth_binary = np.logical_or(truth_binary, self.mask == MASK_OUTSIDE)
        return truth_binary

    @lazy_property.LazyProperty
    def test_binary(self):
        test_binary = self.test != BG
        if self.mask is not None:
            test_binary = np.logical_or(test_binary, self.mask == MASK_OUTSIDE)
        return test_binary

    @lazy_property.LazyProperty
    def test_edt(self):
        test_edt = scipy.ndimage.distance_transform_edt(self.test_mask, self.sampling)
        return test_edt

    @lazy_property.LazyProperty
    def truth_edt(self):
        truth_edt = scipy.ndimage.distance_transform_edt(self.truth_mask, self.sampling)
        return truth_edt

    @lazy_property.LazyProperty
    def truth_bdy(self):
        truth_bdy = find_boundaries_masked(self.truth, mask=self.mask)
        return truth_bdy

    @lazy_property.LazyProperty
    def test_bdy(self):
        test_bdy = find_boundaries_masked(self.test, mask=self.mask)
        return test_bdy

    @lazy_property.LazyProperty
    def signed_truth_edt(self):
        inner_distance = scipy.ndimage.distance_transform_edt(
            scipy.ndimage.binary_erosion(
                self.truth_binary,
                border_value=1,
                structure=scipy.ndimage.generate_binary_structure(
                    self.truth_binary.ndim, self.truth_binary.ndim
                ),
            ),
            sampling=self.sampling,
        )
        outer_distance = scipy.ndimage.distance_transform_edt(
            self.truth_mask, sampling=self.sampling
        )
        return inner_distance - outer_distance

    @lazy_property.LazyProperty
    def signed_test_edt(self):
        inner_distance = scipy.ndimage.distance_transform_edt(
            scipy.ndimage.binary_erosion(
                self.test_binary,
                border_value=1,
                structure=scipy.ndimage.generate_binary_structure(
                    self.test_binary.ndim, self.test_binary.ndim
                ),
            ),
            sampling=self.sampling,
        )
        outer_distance = scipy.ndimage.distance_transform_edt(
            self.test_mask, sampling=self.sampling
        )
        return inner_distance - outer_distance

    @lazy_property.LazyProperty
    def false_positive_distances(self):
        false_positive_distances = self.truth_edt[np.logical_not(self.test_mask)]
        return false_positive_distances

    @lazy_property.LazyProperty
    def false_positive_bdy_distances(self):
        false_positive_bdy_distances = self.signed_truth_edt[self.test_bdy]
        return false_positive_bdy_distances

    @lazy_property.LazyProperty
    def false_positives_with_tolerance(self):
        if self.tol_distance is None:
            raise ValueError("For metrics with tolerance, tol_distance cannot be None.")
        return np.sum(self.false_positive_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_positive_rate_with_tolerance(self):
        if self.mask is not None:
            condition_negative = np.sum(np.logical_and(self.truth_mask, self.mask))
        else:
            condition_negative = np.sum(self.truth_mask)
        return float(self.false_positives_with_tolerance) / float(condition_negative)

    @lazy_property.LazyProperty
    def false_negatives_with_tolerance(self):
        if self.tol_distance is None:
            raise ValueError("For metrics with tolerance, tol_distance cannot be None.")
        return np.sum(self.false_negative_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_negative_bdy_distances(self):
        false_negative_bdy_distances = self.signed_test_edt[self.truth_bdy]
        return false_negative_bdy_distances

    @lazy_property.LazyProperty
    def false_negative_rate_with_tolerance(self):
        condition_positive = len(self.false_negative_distances)
        return float(self.false_negatives_with_tolerance) / float(condition_positive)

    @lazy_property.LazyProperty
    def true_positives_with_tolerance(self):
        all_pos = np.sum(np.invert(self.test_mask & self.truth_mask))
        return (
            all_pos
            - self.false_negatives_with_tolerance
            - self.false_positives_with_tolerance
        )

    @lazy_property.LazyProperty
    def precision_with_tolerance(self):
        return float(self.true_positives_with_tolerance) / float(
            self.true_positives_with_tolerance + self.false_positives_with_tolerance
        )

    @lazy_property.LazyProperty
    def recall_with_tolerance(self):
        return float(self.true_positives_with_tolerance) / float(
            self.true_positives_with_tolerance + self.false_negatives_with_tolerance
        )

    @lazy_property.LazyProperty
    def f1_score_with_tolerance(self):
        if self.recall_with_tolerance == 0 and self.precision_with_tolerance == 0:
            return np.nan
        else:
            return (
                2
                * (self.recall_with_tolerance * self.precision_with_tolerance)
                / (self.recall_with_tolerance + self.precision_with_tolerance)
            )

    @lazy_property.LazyProperty
    def mean_false_positive_distances_clipped(self):
        if self.clip_distance is None:
            raise ValueError(
                "For metrics with clipped distances, clip_distance cannot be None."
            )
        mean_false_positive_distance_clipped = np.mean(
            np.clip(self.false_positive_distances, None, self.clip_distance)
        )
        return mean_false_positive_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_negative_distances_clipped(self):
        if self.clip_distance is None:
            raise ValueError(
                "For metrics with clipped distances, clip_distance cannot be None."
            )
        mean_false_negative_distance_clipped = np.mean(
            np.clip(self.false_negative_distances, None, self.clip_distance)
        )
        return mean_false_negative_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_positive_distance(self):
        mean_false_positive_distance = np.mean(self.false_positive_distances)
        return mean_false_positive_distance

    @lazy_property.LazyProperty
    def mean_false_positive_bdy_distance(self):
        mean_false_positive_bdy_distance = np.mean(
            np.abs(self.false_positive_bdy_distances)
        )
        # return np.sum(self.truth_bdy)
        return mean_false_positive_bdy_distance

    @lazy_property.LazyProperty
    def false_negative_distances(self):
        false_negative_distances = self.test_edt[np.logical_not(self.truth_mask)]
        return false_negative_distances

    @lazy_property.LazyProperty
    def mean_false_negative_distance(self):
        mean_false_negative_distance = np.mean(self.false_negative_distances)
        return mean_false_negative_distance

    @lazy_property.LazyProperty
    def mean_false_negative_bdy_distance(self):
        mean_false_negative_bdy_distance = np.mean(
            np.abs(self.false_negative_bdy_distances)
        )
        # return np.sum(self.test_bdy)
        return mean_false_negative_bdy_distance

    @lazy_property.LazyProperty
    def mean_false_distance(self):
        mean_false_distance = 0.5 * (
            self.mean_false_positive_distance + self.mean_false_negative_distance
        )
        return mean_false_distance

    @lazy_property.LazyProperty
    def mean_false_bdy_distance(self):
        mean_false_bdy_distance = 0.5 * (
            self.mean_false_positive_bdy_distance
            + self.mean_false_negative_bdy_distance
        )
        return mean_false_bdy_distance

    @lazy_property.LazyProperty
    def mean_false_distance_clipped(self):
        mean_false_distance_clipped = 0.5 * (
            self.mean_false_positive_distances_clipped
            + self.mean_false_negative_distances_clipped
        )
        return mean_false_distance_clipped
