import numpy as np
import pytest
import skimage.draw
from CNNectome.validation.organelles import segmentation_metrics
from CNNectome.validation.organelles.segmentation_metrics import *


class TestEvaluator:
    @pytest.mark.parametrize("metric", EvaluationMetrics)
    def test_mask(self, metric):

        # truth_binary = np.ones((20,20,20))
        # test_binary =  np.ones((20,20,20))

        truth_binary_masked, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        truth_binary_masked = truth_binary_masked != 255
        truth_binary_masked = np.stack(
            [
                truth_binary_masked,
            ]
            * 100
        )
        test_binary_masked, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        test_binary_masked = test_binary_masked != 255
        test_binary_masked = np.stack(
            [
                test_binary_masked,
            ]
            * 100
        )
        mask = np.zeros((100, 100, 100))

        mid_slice = (slice(10, -10, None),) * 3

        truth_binary = truth_binary_masked[mid_slice]
        test_binary = test_binary_masked[mid_slice]
        mask[mid_slice] = np.ones(mask[mid_slice].shape)

        metric_params = {"clip_distance": 10, "tol_distance": 10}
        evaluator_unmasked = Evaluator(
            truth_binary,
            test_binary,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=None,
        )
        evaluator_masked = Evaluator(
            truth_binary_masked,
            test_binary_masked,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=mask,
        )

        score_unmasked = evaluator_unmasked.compute_score(metric)
        score_masked = evaluator_masked.compute_score(metric)

        if np.isnan(score_unmasked):
            assert np.isnan(score_masked)
        else:
            assert score_unmasked == score_masked

    @pytest.mark.parametrize(
        "metric_1to2,metric_2to1",
        [
            ("dice", "dice"),
            ("f1_score_with_tolerance", "f1_score_with_tolerance"),
            ("f1_score", "f1_score"),
            ("false_discovery_rate", "false_negative_rate"),
            ("hausdorff", "hausdorff"),
            ("jaccard", "jaccard"),
            ("mean_false_bdy_distance", "mean_false_bdy_distance"),
            ("mean_false_distance_clipped", "mean_false_distance_clipped"),
            ("mean_false_negative_distance", "mean_false_positive_distance"),
            ("mean_false_negative_bdy_distance", "mean_false_positive_bdy_distance"),
            ("mean_false_distance", "mean_false_distance"),
            (
                "mean_false_negative_distance_clipped",
                "mean_false_positive_distance_clipped",
            ),
            ("recall_with_tolerance", "precision_with_tolerance"),
            ("voi", "voi"),
            ("recall", "precision"),
        ],
    )
    def test_symmetry(self, metric_1to2, metric_2to1):
        arr1_binary_masked, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        arr1_binary_masked = arr1_binary_masked != 255
        arr1_binary_masked = np.stack(
            [
                arr1_binary_masked,
            ]
            * 100
        )
        arr2_binary_masked, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        arr2_binary_masked = arr2_binary_masked != 255
        arr2_binary_masked = np.stack(
            [
                arr2_binary_masked,
            ]
            * 100
        )
        mask = np.zeros((100, 100, 100))

        mid_slice = (slice(10, -10, None),) * 3

        mask[mid_slice] = np.ones(mask[mid_slice].shape)

        metric_params = {"clip_distance": 10, "tol_distance": 10}
        evaluator_1to2 = Evaluator(
            arr1_binary_masked,
            arr2_binary_masked,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=mask,
        )
        evaluator_2to1 = Evaluator(
            arr2_binary_masked,
            arr1_binary_masked,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=mask,
        )

        score_1to2 = evaluator_1to2.compute_score(metric_1to2)
        score_2to1 = evaluator_2to1.compute_score(metric_2to1)

        if np.isnan(score_1to2):
            assert np.isnan(score_2to1)
        else:
            assert score_1to2 == score_2to1

    @pytest.mark.parametrize("metric", EvaluationMetrics)
    def test_perfect_score(self, metric):
        truth_binary, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        truth_binary = truth_binary != 255
        truth_binary = np.stack(
            [
                truth_binary,
            ]
            * 100
        )
        test_binary = truth_binary.copy()
        metric_params = {"clip_distance": 20, "tol_distance": 30}
        evaluator = Evaluator(
            truth_binary,
            test_binary,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=None,
        )
        limit_idx = 0 if segmentation_metrics.sorting(metric) == 1 else 1
        print(limit_idx)
        limits = segmentation_metrics.limits(metric)
        print(limits)
        perfect_score = limits[limit_idx]
        print(perfect_score)
        # perfect_score = segmentation_metrics.best(metric)(segmentation_metrics.limits(metric))
        assert evaluator.compute_score(metric) == perfect_score

    @pytest.mark.parametrize("metric", EvaluationMetrics)
    def test_mask_none(self, metric):
        truth_binary, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        test_binary, _ = skimage.draw.random_shapes(
            (100, 100),
            7,
            1,
            allow_overlap=True,
            multichannel=False,
            num_channels=1,
            min_size=7,
            shape="ellipse",
        )
        truth_binary = truth_binary != 255
        test_binary = test_binary != 255
        truth_binary = np.stack(
            [
                truth_binary,
            ]
            * 100
        )
        test_binary = np.stack(
            [
                test_binary,
            ]
            * 100
        )
        mask = np.ones((100, 100, 100))
        metric_params = {"clip_distance": 20, "tol_distance": 30}
        evaluator_with_mask = Evaluator(
            truth_binary,
            test_binary,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=mask,
        )
        evaluator_with_mask_none = Evaluator(
            truth_binary,
            test_binary,
            False,
            False,
            metric_params,
            resolution=(1, 1, 1),
            mask=None,
        )

        score_with_mask_none = evaluator_with_mask_none.compute_score(metric)
        score_with_mask = evaluator_with_mask.compute_score(metric)

        if np.isnan(score_with_mask):
            assert np.isnan(score_with_mask_none)
        else:
            assert score_with_mask == score_with_mask_none

    @pytest.mark.parametrize("array", ["truth", "test", "mask"])
    def test_nonbinary_exception(self, array):
        truth = np.ones((100, 100, 100))
        test = np.ones((100, 100, 100))
        mask = None
        if array == "truth":
            truth = np.random.randint(0, 20, (100, 100, 100))
        elif array == "test":
            test = np.random.randint(0, 20, (100, 100, 100))
        elif array == "mask":
            mask = np.random.randint(0, 20, (100, 100, 100))
        metric_params = dict()
        with pytest.raises(EvaluatorError):
            Evaluator(truth, test, False, False, metric_params, (1, 1, 1), mask)
