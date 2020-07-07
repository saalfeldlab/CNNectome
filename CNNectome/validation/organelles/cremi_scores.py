import numpy as np
import scipy.ndimage
import lazy_property
BG = 0


class CremiEvaluator(object):
    def __init__(self, truth, test, sampling=(1, 1, 1), clip_distance=200, tol_distance=40):
        self.test = test
        self.truth = truth
        self.sampling = sampling
        self.clip_distance = clip_distance
        self.tol_distance = tol_distance

    @lazy_property.LazyProperty
    def test_mask(self):
        # todo: more involved masking
        test_mask = self.test == BG
        return test_mask

    @lazy_property.LazyProperty
    def truth_mask(self):
        truth_mask = self.truth == BG
        return truth_mask

    @lazy_property.LazyProperty
    def test_edt(self):
        test_edt = scipy.ndimage.distance_transform_edt(self.test_mask, self.sampling)
        return test_edt

    @lazy_property.LazyProperty
    def truth_edt(self):
        truth_edt = scipy.ndimage.distance_transform_edt(self.truth_mask, self.sampling)
        return truth_edt

    @lazy_property.LazyProperty
    def false_positive_distances(self):
        test_bin = np.invert(self.test_mask)
        false_positive_distances = self.truth_edt[test_bin]
        return false_positive_distances

    @lazy_property.LazyProperty
    def false_positives_with_tolerance(self):
        return np.sum(self.false_positive_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_positive_rate_with_tolerance(self):
        condition_negative = np.sum(self.truth_mask)
        return float(self.false_positives_with_tolerance) / float(condition_negative)

    @lazy_property.LazyProperty
    def false_negatives_with_tolerance(self):
        return np.sum(self.false_negative_distances > self.tol_distance)

    @lazy_property.LazyProperty
    def false_negative_rate_with_tolerance(self):
        condition_positive = len(self.false_negative_distances)
        return float(self.false_negatives_with_tolerance)/float(condition_positive)

    @lazy_property.LazyProperty
    def true_positives_with_tolerance(self):
        all_pos = np.sum(np.invert(self.test_mask & self.truth_mask))
        return all_pos - self.false_negatives_with_tolerance - self.false_positives_with_tolerance

    @lazy_property.LazyProperty
    def precision_with_tolerance(self):
        return float(self.true_positives_with_tolerance)/float(self.true_positives_with_tolerance + self.false_positives_with_tolerance)

    @lazy_property.LazyProperty
    def recall_with_tolerance(self):
        return float(self.true_positives_with_tolerance)/float(self.true_positives_with_tolerance + self.false_negatives_with_tolerance)

    @lazy_property.LazyProperty
    def f1_score_with_tolerance(self):
        if self.recall_with_tolerance == 0 and self.precision_with_tolerance == 0:
            return np.nan
        else:
            return 2 * (self.recall_with_tolerance * self.precision_with_tolerance) / (self.recall_with_tolerance + self.precision_with_tolerance)

    @lazy_property.LazyProperty
    def mean_false_positive_distances_clipped(self):
        mean_false_positive_distance_clipped = np.mean(np.clip(self.false_positive_distances, None, self.clip_distance))
        return mean_false_positive_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_negative_distances_clipped(self):
        mean_false_negative_distance_clipped = np.mean(np.clip(self.false_negative_distances, None, self.clip_distance))
        return mean_false_negative_distance_clipped

    @lazy_property.LazyProperty
    def mean_false_positive_distance(self):
        mean_false_positive_distance = np.mean(self.false_positive_distances)
        return mean_false_positive_distance

    @lazy_property.LazyProperty
    def false_negative_distances(self):
        truth_bin = np.invert(self.truth_mask)
        false_negative_distances = self.test_edt[truth_bin]
        return false_negative_distances

    @lazy_property.LazyProperty
    def mean_false_negative_distance(self):
        mean_false_negative_distance = np.mean(self.false_negative_distances)
        return mean_false_negative_distance

    @lazy_property.LazyProperty
    def mean_false_distance(self):
        mean_false_distance = 0.5 * (self.mean_false_positive_distance + self.mean_false_negative_distance)
        return mean_false_distance

    @lazy_property.LazyProperty
    def mean_false_distance_clipped(self):
        mean_false_distance_clipped = 0.5 * (self.mean_false_positive_distances_clipped + self.mean_false_negative_distances_clipped)
        return mean_false_distance_clipped