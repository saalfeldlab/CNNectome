import numpy as np
from scipy import ndimage
BG = 0


class CREMIEvaluator:
    def __init__(self, truth, test, sampling=(1, 1, 1), sat_thr=200):
        self.test = test
        self.truth = truth
        self.test_mask = None
        self.truth_mask = None
        self.test_edt = None
        self.truth_edt = None
        self.false_positive_distances = None
        self.false_negative_distances = None
        self.mean_false_positive_distance = None
        self.mean_false_negative_distance = None
        self.mean_false_positive_distance_saturated = None
        self.mean_false_negative_distance_saturated = None
        self.cremi_score = None
        self.cremi_score_saturated = None
        self.sampling = sampling
        self.sat_thr = sat_thr

    def get_test_mask(self):
        # todo: more involved masking
        if self.test_mask is None:
            self.test_mask = self.test == BG
        return self.test_mask

    def get_truth_mask(self):
        if self.truth_mask is None:
            self.truth_mask = self.truth == BG
        return self.truth_mask

    def get_test_edt(self):
        if self.test_edt is None:
            self.test_edt = scipy.ndimage.distance_transform_edt(self.get_test_mask(), self.sampling)
        return self.test_edt

    def get_truth_edt(self):
        if self.truth_edt is None:
            self.truth_edt = scipy.ndimage.distance_transform_edt(self.get_truth_mask(), self.sampling)
        return self.truth_edt

    def get_false_positive_distances(self):
        if self.false_positive_distances is None:
            mask = np.invert(self.get_test_mask())
            self.false_positive_distances = self.get_truth_edt()[mask]
        return self.false_positive_distances

    def get_mean_false_positive_distances_saturated(self):
        if self.mean_false_positive_distance_saturated is None:
            self.mean_false_positive_distance_saturated = np.mean(np.clip(self.get_false_positive_distances(), None, self.sat_thr))
        return self.mean_false_positive_distance_saturated

    def get_mean_false_negative_distances_saturated(self):
        if self.mean_false_negative_distance_saturated is None:
            self.mean_false_negative_distance_saturated = np.mean(np.clip(self.get_false_negative_distances(), None, self.sat_thr))
        return self.mean_false_negative_distance_saturated

    def get_mean_false_positive_distance(self):
        if self.mean_false_positive_distance is None:
            self.mean_false_positive_distance = np.mean(self.get_false_positive_distances())
        return self.mean_false_positive_distance

    def get_false_negative_distances(self):
        if self.false_negative_distances is None:
            mask = np.invert(self.get_truth_mask())
            self.false_negative_distances = self.get_test_edt()[mask]
        return self.false_negative_distances

    def get_mean_false_negative_distance(self):
        if self.mean_false_negative_distance is None:
            self.mean_false_negative_distance = np.mean(self.get_false_negative_distances())
        return self.mean_false_negative_distance

    def get_cremi_score(self):
        if self.cremi_score is None:
            self.cremi_score = 0.5 * (self.get_mean_false_positive_distance() + self.get_mean_false_negative_distance())
        return self.cremi_score

    def get_cremi_score_saturated(self):
        if self.cremi_score_saturated is None:
            self.cremi_score_saturated = 0.5 * (self.get_mean_false_positive_distances_saturated() + self.get_mean_false_negative_distances_saturated())
        return self.cremi_score_saturated