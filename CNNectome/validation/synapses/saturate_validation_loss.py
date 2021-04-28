import h5py
import numpy as np
from scipy import ndimage
import itertools
import json
import os
import sys
import zarr
from CNNectome.utils import config_loader

class Clefts:
    def __init__(self, to_be_evaluated, gt, inverted_mask, scale=200.0):
        self.scale = scale
        test_clefts = to_be_evaluated
        truth_clefts = gt
        self.truth_clefts_invalid = np.logical_or(
            truth_clefts == 0xFFFFFFFFFFFFFFFE, truth_clefts == 0xFFFFFFFFFFFFFFFD
        )
        # print(np.sum(truth_clefts == 0xffffffffffffffff), np.sum(self.truth_clefts_invalid), np.sum(inverted_mask))
        self.truth_clefts_mask = np.logical_or(
            np.logical_or(
                truth_clefts == 0xFFFFFFFFFFFFFFFF, self.truth_clefts_invalid
            ),
            inverted_mask,
        )
        # print(np.sum(self.truth_clefts_mask), np.sum(np.logical_not(self.truth_clefts_mask)))
        # print(np.sum(test_clefts == 0), np.sum(self.truth_clefts_invalid), np.sum(inverted_mask))
        self.test_clefts_mask = np.logical_or(
            np.logical_or(test_clefts == 0, self.truth_clefts_invalid), inverted_mask
        )
        # print(np.sum(self.test_clefts_mask), np.sum(np.logical_not(self.test_clefts_mask)))
        self.test_clefts_edt = ndimage.distance_transform_edt(
            self.test_clefts_mask, sampling=(40.0, 4.0, 4.0)
        )
        self.test_clefts_edt = np.tanh(self.test_clefts_edt / self.scale)
        self.truth_clefts_edt = ndimage.distance_transform_edt(
            self.truth_clefts_mask, sampling=(40.0, 4.0, 4.0)
        )
        self.truth_clefts_edt = np.tanh(self.truth_clefts_edt / self.scale)

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


def bbox2_ND(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(list(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def run_evaluation(experiment_name, scale=200):

    for sample in ["A", "B", "C"]:
        data_path = config_loader.get_config()["synapses"]["cremi17_data_path"]
        setups_path = config_loader.get_config()["synapses"]["training_setups_path"]
        truth_path = os.path.join(data_path, "sample_{0:}_20160501.aligned.uncompressed.hdf".format(
            sample
        ))
        truth_ds = "volumes/labels/clefts"
        mask_path = os.path.join(data_path, "sample_{0:}_cleftsorig_withvalidation.n5".format(
            sample
        ))
        val_ds = "volumes/masks/validation"
        train_ds = "volumes/masks/training"
        truth = h5py.File(truth_path, "r")[truth_ds]
        mask_val = zarr.open(mask_path, mode="r")[val_ds]
        mask_val = np.array(mask_val[:].astype(np.bool))
        x_min, x_max, y_min, y_max, z_min, z_max = bbox2_ND(mask_val)
        s_val = np.s_[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        truth_val = truth[s_val]
        mask_val = mask_val[s_val]

        # mask_train = np.array(mask_train[:].astype(np.bool))
        # x_min, x_max, y_min, y_max, z_min, z_max = bbox2_ND(mask_train)
        # s_train = np.s_[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        # truth_train = truth[s_train]
        # mask_train = mask_train[s_train]
        del truth
        if experiment_name != "DTU2_Bonly":
            iterations = list(range(2000, 84000, 2000))
        else:
            iterations = list(range(2000, 56000, 2000))
        for iteration in iterations:
            validation_json = (os.path.join(setups_path, "miccai_experiments/{0:}/{1:}.n5/it_{"
                "2:}/validation_saturated_s{3:}.json".format(
                    experiment_name, sample, iteration, scale
                )
            ))
            if os.path.exists(validation_json):
                continue
            else:
                test_path = os.path.join(setups_path, "miccai_experiments/{0:}/{1:}.n5".format(
                    experiment_name, sample
                ))
                test_ds = "it_{0:}".format(iteration)
                print(test_path, test_ds)
                test = zarr.open(test_path, mode="r")[test_ds]
                thr = 127
                test_val = test[s_val]
                test_val = test_val > thr
                print("VALIDATION")
                print(
                    "quick shape test", test_val.shape, truth_val.shape, mask_val.shape
                )
                cleft_eval = Clefts(
                    test_val, truth_val, np.logical_not(mask_val), scale=scale
                )
                v_res = dict()
                v_res["v_fn"] = cleft_eval.count_false_negatives()
                v_res["v_fp"] = cleft_eval.count_false_positives()
                v_res["v_df"] = cleft_eval.acc_false_negatives()
                v_res["v_dgt"] = cleft_eval.acc_false_positives()
                print(
                    "fn",
                    v_res["v_fn"],
                    v_res["v_df"],
                    "fp",
                    v_res["v_fp"],
                    v_res["v_dgt"],
                )

                with open(validation_json, "w") as f:
                    json.dump(v_res, f)
                del test_val, v_res
            # print("TRAINING")
            # test_train = test[s_train]
            # test_train = (test_train > thr)
            # cleft_eval = Clefts(test_train, truth_train, np.logical_not(mask_train))
            # t_res = dict()
            # t_res['t_fn'] = cleft_eval.count_false_negatives()
            # t_res['t_fp'] = cleft_eval.count_false_positives()
            # t_res['t_df'] = cleft_eval.acc_false_negatives()
            # t_res['t_dgt'] = cleft_eval.acc_false_positives()
            # print('fn', t_res['t_fn'],  t_res['t_df'], 'fp', t_res['t_fp'], t_res['t_dgt'])
            # training_json ='/nrs/saalfeld/heinrichl/synapses/miccai_experiments/{0:}/{1:}.n5/it_{' \
            #                 '2:}/training.json'.format(experiment_name, sample, iteration)
            # with open(training_json, 'w') as f:
            #    json.dump(t_res, f)
            # del test_train, t_res, test
