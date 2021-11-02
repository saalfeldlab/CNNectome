import pytest
import numpy as np
import skimage.draw

from CNNectome.validation.organelles.segmentation_metrics import *
class TestEvaluator():
    @pytest.mark.parametrize('metric', EvaluationMetrics)
    def test_mask(self, metric):
    
        #truth_binary = np.ones((20,20,20))
        #test_binary =  np.ones((20,20,20))
        
        truth_binary_masked, _ = skimage.draw.random_shapes((100,100),7,1,allow_overlap=True, multichannel=False,num_channels=1, min_size=7, shape='ellipse')
        truth_binary_masked = truth_binary_masked != 255
        truth_binary_masked = np.stack([truth_binary_masked,]*100)
        test_binary_masked, _ = skimage.draw.random_shapes((100,100),7,1,allow_overlap=True, multichannel=False,num_channels=1, min_size=7, shape='ellipse')
        test_binary_masked = test_binary_masked != 255
        test_binary_masked = np.stack([test_binary_masked,]*100)
        mask = np.zeros((100,100,100))
        
        mid_slice = (slice(10,-10,None),)*3
        
        truth_binary = truth_binary_masked[mid_slice]
        test_binary = test_binary_masked[mid_slice]
        mask[mid_slice] = np.ones(mask[mid_slice].shape)

        metric_params = {"clip_distance": 10, "tol_distance":10}
        evaluator_unmasked = Evaluator(truth_binary, test_binary, False, False, metric_params, resolution=(1,1,1), mask=None)
        evaluator_masked = Evaluator(truth_binary_masked, test_binary_masked, False, False, metric_params, resolution=(1,1,1), mask=mask)

        score_unmasked = evaluator_unmasked.compute_score(metric)
        score_masked = evaluator_masked.compute_score(metric)
        print(score_unmasked)
        if np.isnan(score_unmasked):
            assert np.isnan(score_masked)
        else:
            assert score_unmasked == score_masked
    
    @pytest.mark.parametrize('metric', EvaluationMetrics)
    def test_mask_none(self, metric):
        truth_binary, _ = skimage.draw.random_shapes((100,100),7,1,allow_overlap=True, multichannel=False, num_channels=1, min_size=7, shape='ellipse')
        test_binary, _ = skimage.draw.random_shapes((100,100),7,1,allow_overlap=True, multichannel=False, num_channels=1, min_size=7, shape='ellipse')
        truth_binary = truth_binary != 255
        test_binary = test_binary != 255
        truth_binary = np.stack([truth_binary,]*100)
        test_binary = np.stack([test_binary,]*100)
        mask = np.ones((100,100,100))
        metric_params = {"clip_distance": 20, "tol_distance": 30}
        evaluator_with_mask = Evaluator(truth_binary, test_binary, False, False, metric_params, resolution=(1,1,1), mask=mask)
        evaluator_with_mask_none = Evaluator(truth_binary, test_binary, False, False, metric_params, resolution=(1,1,1), mask=None)

        score_with_mask_none = evaluator_with_mask_none.compute_score(metric)
        score_with_mask = evaluator_with_mask.compute_score(metric)
        

        if np.isnan(score_with_mask):
            assert np.isnan(score_with_mask_none)
        else:
            assert score_with_mask == score_with_mask_none
    
    @pytest.mark.parametrize('array', ['truth', 'test', 'mask'])
    def test_nonbinary_exception(self, array):
        truth = np.ones((100,100,100))
        test = np.ones((100,100,100))
        mask = None
        if array == "truth":
            truth = np.random.randint(0, 20, (100,100,100))
        elif array == "test":
            test = np.random.randint(0, 20, (100,100,100))
        elif array == "mask":
            mask = np.random.randint(0, 20, (100,100,100))
        metric_params = dict()
        with pytest.raises(EvaluatorError):
            Evaluator(truth, test, False, False, metric_params, (1,1,1), mask)

        

