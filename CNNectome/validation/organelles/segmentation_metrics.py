from CNNectome.validation.organelles import cremi_scores
from enum import Enum
import numpy as np
import SimpleITK as sitk
import lazy_property
import cremi.evaluation


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


def best(argument):
    switcher = {
        EvaluationMetrics.dice:                                 np.nanargmax,
        EvaluationMetrics.jaccard:                              np.nanargmax,
        EvaluationMetrics.hausdorff:                            np.nanargmin,
        EvaluationMetrics.false_negative_rate:                  np.nanargmin,
        EvaluationMetrics.false_negative_rate_with_tolerance:   np.nanargmin,
        EvaluationMetrics.false_positive_rate:                  np.nanargmin,
        EvaluationMetrics.false_discovery_rate:                 np.nanargmin,
        EvaluationMetrics.false_positive_rate_with_tolerance:   np.nanargmin,
        EvaluationMetrics.voi:                                  np.nanargmin,
        EvaluationMetrics.mean_false_distance:                  np.nanargmin,
        EvaluationMetrics.mean_false_positive_distance:         np.nanargmin,
        EvaluationMetrics.mean_false_negative_distance:         np.nanargmin,
        EvaluationMetrics.mean_false_distance_clipped:          np.nanargmin,
        EvaluationMetrics.mean_false_negative_distance_clipped: np.nanargmin,
        EvaluationMetrics.mean_false_positive_distance_clipped: np.nanargmin,
        EvaluationMetrics.precision_with_tolerance:             np.nanargmax,
        EvaluationMetrics.recall_with_tolerance:                np.nanargmax,
        EvaluationMetrics.f1_score_with_tolerance:              np.nanargmax,
        EvaluationMetrics.precision:                            np.nanargmax,
        EvaluationMetrics.recall:                               np.nanargmax,
        EvaluationMetrics.f1_score:                             np.nanargmax
    }
    return switcher.get(argument)


class Evaluator(object):
    def __init__(self, truth_binary, test_binary, truth_empty, test_empty, metric_params, resolution):
        self.truth = truth_binary.astype(np.uint8)
        self.test = test_binary.astype(np.uint8)
        self.truth_empty = truth_empty
        self.test_empty = test_empty
        self.cremieval = cremi_scores.CremiEvaluator(truth_binary, test_binary,
                                                     sampling=resolution,
                                                     clip_distance=metric_params['clip_distance'],
                                                     tol_distance=metric_params['tol_distance'])
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
            return None

    def jaccard(self):
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetJaccardCoefficient()
        else:
            return None

    def hausdorff(self):
        if self.truth_empty and self.test_empty:
            return 0
        elif not self.truth_empty and not self.test_empty:
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(self.test_itk, self.truth_itk)
            return hausdorff_distance_filter.GetHausdorffDistance()
        else:
            return np.Inf

    def false_negative_rate(self):
        if self.truth_empty and self.pred_emtpy:
            return 0
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetFalseNegativeError()
        else:
            return None

    def false_positive_rate(self):
        return (self.false_discovery_rate() * np.sum(self.test != 0)) / np.sum(self.truth == 0)

    def false_discovery_rate(self):
        if self.test_empty:
            return 0
        if (not self.truth_empty) or (not self.test_empty):
            return self.overlap_measures_filter.GetFalsePositiveError()
        else:
            return None

    def precision(self):
        pred_pos = np.sum(self.test != 0)
        tp = pred_pos - (self.false_discovery_rate() * pred_pos)
        return float(tp)/float(pred_pos)

    def recall(self):
        cond_pos = np.sum(self.truth != 0)
        tp = cond_pos - (self.false_negative_rate() * cond_pos)
        return float(tp)/float(cond_pos)

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (rec * prec) / (rec + prec)

    def voi(self):
        voi_split, voi_merge = cremi.evaluation.voi(self.test + 1, self.truth + 1, ignore_groundtruth=[])
        return voi_split + voi_merge

    def mean_false_distance(self):
        return self.cremieval.mean_false_distance

    def mean_false_negative_distance(self):
        return self.cremieval.mean_false_negative_distance

    def mean_false_positive_distance(self):
        return self.cremieval.mean_false_positive_distance

    def mean_false_distance_clipped(self):
        return self.cremieval.mean_false_distance_clipped

    def mean_false_negative_distance_clipped(self):
        return self.cremieval.mean_false_negative_distances_clipped

    def mean_false_positive_distance_clipped(self):
        return self.cremieval.mean_false_positive_distances_clipped

    def false_positive_rate_with_tolerance(self):
        return self.cremieval.false_positive_rate_with_tolerance

    def false_negative_rate_with_tolerance(self):
        return self.cremieval.false_negative_rate_with_tolerance

    def precision_with_tolerance(self):
        return self.cremieval.precision_with_tolerance

    def recall_with_tolerance(self):
        return self.cremieval.recall_with_tolerance

    def f1_score_with_tolerance(self):
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



