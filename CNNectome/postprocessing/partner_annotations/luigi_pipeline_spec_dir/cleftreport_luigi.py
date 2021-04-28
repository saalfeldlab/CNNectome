import luigi
import os
import numpy as np
from scipy import ndimage
import h5py
import json
import zarr
import numcodecs
from CNNectome.utils import config_loader
from threshold_luigi import Threshold

# d OUTSIDE
# e INVALID
# f TRANSPARENT
class Clefts:
    def __init__(self, to_be_evaluated, gt, inverted_mask):
        test_clefts = to_be_evaluated
        truth_clefts = gt
        self.truth_clefts_invalid = np.logical_or(
            truth_clefts == 0xFFFFFFFFFFFFFFFE, truth_clefts == 0xFFFFFFFFFFFFFFFD
        )

        self.truth_clefts_mask = np.logical_or(
            np.logical_or(
                truth_clefts == 0xFFFFFFFFFFFFFFFF, self.truth_clefts_invalid
            ),
            inverted_mask,
        )

        self.test_clefts_mask = np.logical_or.reduce(
            (test_clefts == 0, self.truth_clefts_invalid, inverted_mask)
        )
        self.test_clefts_edt = ndimage.distance_transform_edt(
            self.test_clefts_mask, sampling=(40.0, 4.0, 4.0)
        )
        self.truth_clefts_edt = ndimage.distance_transform_edt(
            self.truth_clefts_mask, sampling=(40.0, 4.0, 4.0)
        )

    def count_false_positives(self, threshold=200):
        mask1 = np.invert(self.test_clefts_mask)
        mask2 = self.truth_clefts_edt > threshold
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)]
        return false_positives.size

    def count_false_negatives(self, threshold=200):
        mask1 = np.invert(self.truth_clefts_mask)
        mask2 = self.test_clefts_edt > threshold
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size

    def acc_false_positives(self):
        mask = np.invert(self.test_clefts_mask)
        false_positives = self.truth_clefts_edt[mask]
        try:
            stats = {
                "mean": np.mean(false_positives),
                "std": np.std(false_positives),
                "max": np.amax(false_positives),
                "count": false_positives.size,
                "median": np.median(false_positives),
            }
        except ValueError:
            assert np.sum(mask) == 0
            stats = {
                "mean": np.mean(false_positives),
                "std": np.std(false_positives),
                "max": 0,
                "count": false_positives.size,
                "median": np.median(false_positives),
            }

        return stats

    def acc_false_negatives(self):
        mask = np.invert(self.truth_clefts_mask)
        false_negatives = self.test_clefts_edt[mask]
        stats = {
            "mean": np.mean(false_negatives),
            "std": np.std(false_negatives),
            "max": np.amax(false_negatives),
            "count": false_negatives.size,
            "median": np.median(false_negatives),
        }
        return stats


class CleftReport(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    de = luigi.Parameter()
    m = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources = {"ram": 10}

    @property
    def priority(self):
        if int(self.it) % 10000 == 0:
            return 1.0 / int(self.it)
        else:
            return 0.0

    def requires(self):
        return Threshold(
            self.it, self.dt, self.aug, self.de, self.samples, self.data_eval
        )

    def output(self):
        cleftrep = os.path.join(
            os.path.dirname(self.input().fn), "cleft." + self.m + ".json"
        )
        return luigi.LocalTarget(cleftrep)

    def run(self):
        progress = 0.0
        self.set_progress_percentage(progress)
        results = dict()
        for s in self.samples:
            thr = 127
            testfile = os.path.join(os.path.dirname(self.input().fn), s + ".n5")
            truthfile = os.path.join(
                config_loader.get_config()["synapses"]["cremieval_path"],
                self.de,
                s + ".n5",
            )
            test = np.array(
                zarr.open(testfile, mode="r")[
                    "clefts_cropped_thr" + str(thr)
                ][:]
            )
            truth = np.array(
                zarr.open(truthfile, mode="r")[
                    "volumes/labels/clefts_cropped"
                ][:]
            )
            mask = np.array(
                zarr.open(truthfile, mode="r")[
                    "volumes/masks/" + self.m + "_cropped"
                ][:]
            )
            clefts_evaluation = Clefts(test, truth, np.logical_not(mask))
            results[s] = dict()
            results[s][
                "false negatives count"
            ] = clefts_evaluation.count_false_negatives()
            results[s][
                "false positives count"
            ] = clefts_evaluation.count_false_positives()
            results[s][
                "false negative distance"
            ] = clefts_evaluation.acc_false_negatives()
            results[s][
                "false positive distance"
            ] = clefts_evaluation.acc_false_positives()
            progress += 100.0 / len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        with self.output().open("w") as done:
            json.dump(results, done)
