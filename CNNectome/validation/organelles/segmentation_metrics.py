from CNNectome.validation.organelles import cremi_scores
from enum import Enum
import numpy as np
import SimpleITK as sitk
import lazy_property
import cremi.evaluation

class EvaluatorError(Exception):
    def __init__(self, array_name, values):
        message = f"Array {array_name:} should only contain values 0, 1 but found {values:}"
        super().__init__(message)
    

class EvaluationMetrics(str, Enum):
    dice = 'dice'
    jaccard = 'jaccard'
    hausdorff = 'hausdorff'
    false_negative_rate = 'false_negative_rate'
    false_negative_rate_with_tolerance = 'false_negative_rate_with_tolerance'
    false_positive_rate = 'false_positive_rate'
    false_discovery_rate = 'false_discovery_rate'
    false_positive_rate_with_tolerance = 'false_positive_rate_with_tolerance'
    voi = 'voi'
    mean_false_distance = 'mean_false_distance'
    mean_false_negative_distance = 'mean_false_negative_distance'
    mean_false_positive_distance = 'mean_false_positive_distance'
    mean_false_distance_clipped = 'mean_false_distance_clipped'
    mean_false_negative_distance_clipped = 'mean_false_negative_distance_clipped'
    mean_false_positive_distance_clipped = 'mean_false_positive_distance_clipped'
    precision_with_tolerance = 'precision_with_tolerance'
    recall_with_tolerance = 'recall_with_tolerance'
    f1_score_with_tolerance = 'f1_score_with_tolerance'
    precision = 'precision'
    recall = 'recall'
    f1_score = 'f1_score'

def display_name(metric):
    switcher = {
        EvaluationMetrics.dice: "F1 Score",
        EvaluationMetrics.jaccard: "Jaccard Index",
        EvaluationMetrics.hausdorff: "Hausdorff Distance",
        EvaluationMetrics.false_negative_rate: "False Negative Rate",
        EvaluationMetrics.false_negative_rate_with_tolerance: "False Negative Rate with Tolerance Distance",
        EvaluationMetrics.false_positive_rate: "False Positive Rate",
        EvaluationMetrics.false_discovery_rate: "False Discovery Rate",
        EvaluationMetrics.false_positive_rate_with_tolerance: "False Positive Rate with Tolerance Distance",
        EvaluationMetrics.voi: "Variation Of Information",
        EvaluationMetrics.mean_false_distance: "Mean False Distance",
        EvaluationMetrics.mean_false_positive_distance: "Mean False Positive Distance",
        EvaluationMetrics.mean_false_negative_distance: "Mean False Negative Distance",
        EvaluationMetrics.mean_false_distance_clipped: "Mean False Distance (Clipped)",
        EvaluationMetrics.mean_false_positive_distance_clipped: "Mean False Positive Distance (Clipped)",
        EvaluationMetrics.mean_false_negative_distance_clipped: "Mean False Negative Distance (Clipped)",
        EvaluationMetrics.precision_with_tolerance: "Precision with Tolerance Distance",
        EvaluationMetrics.recall_with_tolerance: "Recall with Tolerance Distance",
        EvaluationMetrics.f1_score_with_tolerance: "F1 Score with Tolerance Distance",
        EvaluationMetrics.precision: "Precision",
        EvaluationMetrics.recall: "Recall",
        EvaluationMetrics.f1_score: "F1 Score"
    }
    return switcher.get(metric)

def filter_params(params, metric):
    params_by_metric = {
        EvaluationMetrics.dice: (),
        EvaluationMetrics.jaccard: (),
        EvaluationMetrics.hausdorff: (),
        EvaluationMetrics.false_negative_rate: (),
        EvaluationMetrics.false_negative_rate_with_tolerance: ('tol_distance',),
        EvaluationMetrics.false_positive_rate: (),
        EvaluationMetrics.false_discovery_rate: (),
        EvaluationMetrics.false_positive_rate_with_tolerance: ('tol_distance',),
        EvaluationMetrics.voi: (),
        EvaluationMetrics.mean_false_distance: (),
        EvaluationMetrics.mean_false_positive_distance: (),
        EvaluationMetrics.mean_false_negative_distance: (),
        EvaluationMetrics.mean_false_distance_clipped: ('clip_distance',),
        EvaluationMetrics.mean_false_negative_distance_clipped: ('clip_distance',),
        EvaluationMetrics.mean_false_positive_distance_clipped: ('clip_distance',),
        EvaluationMetrics.precision_with_tolerance: ('tol_distance',),
        EvaluationMetrics.recall_with_tolerance: ('tol_distance',),
        EvaluationMetrics.f1_score_with_tolerance: ('tol_distance',),
        EvaluationMetrics.precision: (),
        EvaluationMetrics.recall: (),
        EvaluationMetrics.f1_score: ()
    }
    return dict((k, v) for k, v in params.items() if k in params_by_metric.get(metric))


def sorting(argument):
    switcher = {
        EvaluationMetrics.dice:                                 -1,
        EvaluationMetrics.jaccard:                              -1,
        EvaluationMetrics.hausdorff:                            1,
        EvaluationMetrics.false_negative_rate:                  1,
        EvaluationMetrics.false_negative_rate_with_tolerance:   1,
        EvaluationMetrics.false_positive_rate:                  1,
        EvaluationMetrics.false_discovery_rate:                 1,
        EvaluationMetrics.false_positive_rate_with_tolerance:   1,
        EvaluationMetrics.voi:                                  1,
        EvaluationMetrics.mean_false_distance:                  1,
        EvaluationMetrics.mean_false_positive_distance:         1,
        EvaluationMetrics.mean_false_negative_distance:         1,
        EvaluationMetrics.mean_false_distance_clipped:          1,
        EvaluationMetrics.mean_false_negative_distance_clipped: 1,
        EvaluationMetrics.mean_false_positive_distance_clipped: 1,
        EvaluationMetrics.precision_with_tolerance:             -1,
        EvaluationMetrics.recall_with_tolerance:                -1,
        EvaluationMetrics.f1_score_with_tolerance:              -1,
        EvaluationMetrics.precision:                            -1,
        EvaluationMetrics.recall:                               -1,
        EvaluationMetrics.f1_score:                             -1
    }
    return switcher.get(argument)


def best(argument):
    switcher = {
        -1: np.nanargmax,
        1: np.nanargmin
    }
    return switcher.get(sorting(argument))


def limits(argument):
    switcher = {
        EvaluationMetrics.dice:                                 (0, 1),
        EvaluationMetrics.jaccard:                              (0, 1),
        EvaluationMetrics.hausdorff:                            (0, None),
        EvaluationMetrics.false_negative_rate:                  (0, 1),
        EvaluationMetrics.false_negative_rate_with_tolerance:   (0, 1),
        EvaluationMetrics.false_positive_rate:                  (0, 1),
        EvaluationMetrics.false_discovery_rate:                 (0, 1),
        EvaluationMetrics.false_positive_rate_with_tolerance:   (0, 1),
        EvaluationMetrics.voi:                                  (0, 1),
        EvaluationMetrics.mean_false_distance:                  (0, None),
        EvaluationMetrics.mean_false_positive_distance:         (0, None),
        EvaluationMetrics.mean_false_negative_distance:         (0, None),
        EvaluationMetrics.mean_false_distance_clipped:          (0, None),
        EvaluationMetrics.mean_false_negative_distance_clipped: (0, None),
        EvaluationMetrics.mean_false_positive_distance_clipped: (0, None),
        EvaluationMetrics.precision_with_tolerance:             (0, 1),
        EvaluationMetrics.recall_with_tolerance:                (0, 1),
        EvaluationMetrics.f1_score_with_tolerance:              (0, 1),
        EvaluationMetrics.precision:                            (0, 1),
        EvaluationMetrics.recall:                               (0, 1),
        EvaluationMetrics.f1_score:                             (0, 1)
    }
    return switcher.get(argument)


class Evaluator(object):
    def __init__(self, truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution, mask):

        self.truth = truth_binary.astype(np.uint8)
        self.test = test_binary.astype(np.uint8)
        self.truth_empty = truth_empty
        self.test_empty = test_empty
        self.mask = mask
        if mask is not None:
            self.mask = self.mask.astype(np.uint8)
            self.truth = (self.truth*mask).astype(np.uint8)
            self.test = (self.test*mask).astype(np.uint8)

        truth_values = set(np.unique(self.truth))
        if not truth_values.issubset({0,1}):
            raise EvaluatorError("truth_array", truth_values)
        test_values = set(np.unique(self.test))
        if not test_values.issubset({0,1}):
            raise EvaluatorError("test_array", test_values)
        if mask is not None:
            mask_values = set(np.unique(self.mask))
            if not mask_values.issubset({0,1}):
                raise EvaluatorError("mask", mask_values)
        self.cremieval = cremi_scores.CremiEvaluator(self.truth, self.test,
                                                     sampling=resolution,
                                                     clip_distance=metric_params['clip_distance'],
                                                     tol_distance=metric_params['tol_distance'],
                                                     mask=self.mask)
        self.resolution = resolution

    @lazy_property.LazyProperty
    def truth_itk(self):
        res = sitk.GetImageFromArray(self.truth)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def test_itk(self):
        res = sitk.GetImageFromArray(self.test)
        res.SetSpacing(self.resolution)
        return res

    @lazy_property.LazyProperty
    def overlap_measures_filter(self):
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(self.test_itk, self.truth_itk)
        return overlap_measures_filter

    def dice(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetDiceCoefficient()
        else:
            return np.nan

    def jaccard(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetJaccardCoefficient()
        else:
            return np.nan

    def hausdorff(self):
        if self.truth_empty and self.test_empty:
            return 0
        elif not self.truth_empty and not self.test_empty:
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(self.test_itk, self.truth_itk)
            return hausdorff_distance_filter.GetHausdorffDistance()
        else:
            return np.nan

    def false_negative_rate(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.overlap_measures_filter.GetFalseNegativeError()

    def false_positive_rate(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            if self.mask is not None:
                negatives = np.sum(self.truth[self.mask!=0] == 0)
            else:
                negatives = np.sum(self.truth == 0)
            return (self.false_discovery_rate() * np.sum(self.test != 0)) / negatives

    def false_discovery_rate(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetFalsePositiveError()
        else:
            return np.nan

    def precision(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            pred_pos = np.sum(self.test != 0)
            tp = pred_pos - (self.false_discovery_rate() * pred_pos)
            return float(tp)/float(pred_pos)

    def recall(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            cond_pos = np.sum(self.truth != 0)
            tp = cond_pos - (self.false_negative_rate() * cond_pos)
            return float(tp)/float(cond_pos)

    def f1_score(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            prec = self.precision()
            rec = self.recall()
            if prec == 0 and rec == 0:
                return np.nan
            else:
                return 2 * (rec * prec) / (rec + prec)

    def voi(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            if self.mask is not None:
                test_voi = self.test + self.mask # add 1 in valid area, keep values of 0 in invalid area
                truth_voi = self.truth + self.mask
            else:
                test_voi = self.test + 1
                truth_voi = self.truth + 1
            voi_split, voi_merge = cremi.evaluation.voi(test_voi, truth_voi, 
                                                        ignore_groundtruth=[0], 
                                                        ignore_reconstruction=[0])
            return voi_split + voi_merge

    def mean_false_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance

    def mean_false_negative_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distance

    def mean_false_positive_distance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distance

    def mean_false_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_distance_clipped

    def mean_false_negative_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_negative_distances_clipped

    def mean_false_positive_distance_clipped(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.mean_false_positive_distances_clipped

    def false_positive_rate_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_positive_rate_with_tolerance

    def false_negative_rate_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.false_negative_rate_with_tolerance

    def precision_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.precision_with_tolerance

    def recall_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.recall_with_tolerance

    def f1_score_with_tolerance(self):
        if self.truth_empty or self.test_empty:
            return np.nan
        else:
            return self.cremieval.f1_score_with_tolerance

    def compute_score(self, argument):
        switcher = {
            EvaluationMetrics.dice: self.dice,
            EvaluationMetrics.jaccard: self.jaccard,
            EvaluationMetrics.hausdorff: self.hausdorff,
            EvaluationMetrics.false_negative_rate: self.false_negative_rate,
            EvaluationMetrics.false_positive_rate: self.false_positive_rate,
            EvaluationMetrics.false_discovery_rate: self.false_discovery_rate,
            EvaluationMetrics.voi: self.voi,
            EvaluationMetrics.mean_false_distance: self.mean_false_distance,
            EvaluationMetrics.mean_false_negative_distance: self.mean_false_negative_distance,
            EvaluationMetrics.mean_false_positive_distance: self.mean_false_positive_distance,
            EvaluationMetrics.mean_false_distance_clipped: self.mean_false_distance_clipped,
            EvaluationMetrics.mean_false_negative_distance_clipped: self.mean_false_negative_distance_clipped,
            EvaluationMetrics.mean_false_positive_distance_clipped: self.mean_false_positive_distance_clipped,
            EvaluationMetrics.false_positive_rate_with_tolerance: self.false_positive_rate_with_tolerance,
            EvaluationMetrics.false_negative_rate_with_tolerance: self.false_negative_rate_with_tolerance,
            EvaluationMetrics.recall_with_tolerance: self.recall_with_tolerance,
            EvaluationMetrics.precision_with_tolerance: self.precision_with_tolerance,
            EvaluationMetrics.f1_score_with_tolerance: self.f1_score_with_tolerance,
            EvaluationMetrics.recall: self.recall,
            EvaluationMetrics.precision: self.precision,
            EvaluationMetrics.f1_score: self.f1_score,
        }
        return switcher.get(argument)()



