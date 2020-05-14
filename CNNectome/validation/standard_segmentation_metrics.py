import numpy as np
import pymongo
import zarr
import scipy.spatial.distance
from CNNectome.utils.label import Label
from CNNectome.validation import cremi_scores
import numcodecs
import SimpleITK as sitk
import sklearn.metrics
import cremi.evaluation
from enum import Enum


class EvaluationMetrics(Enum):
    dice = 0
    jaccard = 1
    hausdorff = 2
    false_negatives = 3
    false_positives = 4
    adjusted_rand_index = 5
    voi = 6
    cremiscore = 7
    cremiscore_saturated = 8


def compute_score(argument, evaluator):
    switcher = {
        EvaluationMetrics.dice:                 evaluator.dice(),
        EvaluationMetrics.jaccard:              evaluator.jaccard(),
        EvaluationMetrics.hausdorff:            evaluator.hausdorff(),
        EvaluationMetrics.false_negatives:      evaluator.false_negatives(),
        EvaluationMetrics.false_positives:      evaluator.false_positives(),
        EvaluationMetrics.adjusted_rand_index:  evaluator.adjusted_rand_index(),
        EvaluationMetrics.voi:                  evaluator.voi(),
        EvaluationMetrics.cremiscore:           evaluator.cremiscore(),
        EvaluationMetrics.cremiscore_saturated: evaluator.cremiscore_saturated()
    }
    return switcher.get(argument)


class Evaluator(object):
    def __init__(self, binary_seg_gt, binary_seg_pred, gt_empty, seg_empty):
        self.overlap_measures_filter = None
        self.A = binary_seg_gt.astype(np.uint8)
        self.B = binary_seg_pred.astype(np.uint8)
        self.A_itk = sitk.GetImageFromArray(self.A)
        self.B_itk = sitk.GetImageFromArray(self.B)
        self.gt_empty = gt_empty
        self.seg_empty = seg_empty
        self.cremieval = cremi_scores.CremiEvaluator(binary_seg_gt, binary_seg_pred)

    def get_overlap_measures_filter(self):
        if self.overlap_measures_filter is None:
            self.overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            self.overlap_measures_filter.Execute(self.A_itk, self.B_itk)
        return self.overlap_measures_filter

    def dice(self):
        if (not self.gt_empty) or (not self.seg_empty):
            return self.get_overlap_measures_filter().GetDiceCoefficient()
        else:
            return None

    def jaccard(self):
        if (not self.gt_empty) or (not self.seg_empty):
            return self.get_overlap_measures_filter().GetJaccardCoefficient()
        else:
            return None

    def hausdorff(self):
        if self.gt_empty and self.seg_empty:
            return 0
        elif not self.gt_empty and not self.seg_empty:
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(self.A_itk, self.B_itk)
            return hausdorff_distance_filter.GetHausdorffDistance()
        else:
            return np.Inf

    def false_negatives(self):
        if (not self.gt_empty) or (not self.seg_empty):
            return self.get_overlap_measures_filter().GetFalseNegativeError()
        else:
            return None

    def false_positives(self):
        if (not self.gt_empty) or (not self.seg_empty):
            return self.get_overlap_measures_filter().GetFalsePositiveError()
        else:
            return None

    def adjusted_rand_index(self):
        return sklearn.metrics.adjusted_rand_score(self.A.flatten(), self.B.flatten())

    def voi(self):
        voi_split, voi_merge = cremi.evaluation.voi(self.B + 1, self.A + 1, ignore_groundtruth=[])
        return voi_split + voi_merge

    def cremi(self):
        return self.cremieval.get_cremi_score()

    def cremi_saturated(self):
        return self.cremieval.get_cremi_score_saturated()


def downsample(arr, factor=2):
    return arr[(slice(None, None, factor),) * arr.ndim]


def get_label_ids_by_category(crop, category):
    return [l[0] for l in crop['labels'][category]]


def get_all_annotated_label_ids(crop):
    return get_label_ids_by_category(crop, "present_annotated") + get_label_ids_by_category(crop, "absent_annotated")


def read_training(crop, label, gt_version = "v0003"):
    n5file = zarr.open(crop["parent"], mode="r")
    blueprint_label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/{{label:}}"
    label_ds = blueprint_label_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    #todo: extract different label as necessary
    gt_seg = n5file[label_ds.format(label="all")]
    gt_seg = downsample(gt_seg)

    blueprint_labelmask_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/masks/{{label:}}"
    labelmask_ds = blueprint_labelmask_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    labelmask_ds.format(label=label.labelname)
    if labelmask_ds in n5file:
        mask = n5file[labelmask_ds]
    else:
        if label.generic_label is not None:
            specific_labels = list(set(label.labelid)-set(label.generic_label))
            generic_condition = (all(l in get_all_annotated_label_ids(crop) for l in label.generic_label) or
                                 all(l in get_all_annotated_label_ids(crop) for l in specific_labels))
        else:
            generic_condition=False
        if all(l in get_all_annotated_label_ids(crop) for l in label.labelid) or generic_condition:
            mask = ((gt_seg > 0) * 1).astype(np.bool)
        else:
            mask = ((gt_seg > 0) * 0).astype(np.bool)

    return gt_seg, mask


def extract_binary_class(gt_seg, label):
    return np.in1d(gt_seg.ravel(), label.labelid).reshape(gt_seg.shape)


def threshold(prediction, thr=128):
    return (prediction>thr).astype(np.bool)


def read_prediction(prediction_path, label, offset, shape):
    n5file = zarr.open(prediction_path, mode="r")
    pred = n5file[label.labelname]
    sl = tuple(slice(int(o), int(o+s), None) for o, s in zip(offset, shape))
    return pred[sl]


def get_offset_and_shape_from_training_crop(crop, gt_version="v0003"):
    n5file = zarr.open(crop["parent"], mode="r")
    label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/all".format(version=gt_version.lstrip("v"), cropno=crop["number"])
    offset_wc = n5file[label_ds].attrs["offset"][::-1]
    offset = tuple(np.array(offset_wc)/4.)
    shape = tuple(np.array(n5file[label_ds].shape)/2.)
    return offset, shape


def get_parent(prediction_path, label):
    n5file = zarr.open(prediction_path, mode="r")
    pred = n5file[label.labelname]
    return pred.attrs["raw_data_path"]


def score_average(score_dict, size_dict):
    avg_score = 0.
    total_size = 0.
    for cropno, score in score_dict.items():
        try:
            avg_score += score * size_dict[cropno]
        except RuntimeWarning:
            print(score, size_dict[cropno])
        total_size += size_dict[cropno]
    avg_score /= total_size
    return avg_score


# def evaluate(label, predictions_to_compare, db_username, db_password, db_name="crops", gt_version="v0003",
#              completion_min=5):
#     # prepare db access
#     client = pymongo.MongoClient("cosem.int.janelia.org:27017", username=db_username, password=db_password)
#     db = client[db_name]  # db_name = "crops"
#     collection = db[gt_version]  # gt_version = "v0003"
#     skip = {"_id": 0, "number": 1, "labels": 1, "parent": 1, "dimensions": 1}
#
#     # prepare sitk filters
#     hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
#     overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
#     dice_res = {}
#     jaccard_res = {}
#     hd_res = {}
#     fn_res = {}
#     fp_res = {}
#     ari_res = {}
#     voi_res = {}
#     for pred_path in predictions_to_compare:
#         parent = get_parent(pred_path, label)
#         filter = {"completion": {"$gte": completion_min}, "parent": parent}
#         dice_res[pred_path] = dict()
#         jaccard_res[pred_path] = dict()
#         hd_res[pred_path] = dict()
#         fn_res[pred_path] = dict()
#         fp_res[pred_path] = dict()
#         ari_res[pred_path] = dict()
#         voi_res[pred_path] = dict()
#
#         crop_sizes = dict()
#         for crop in collection.find(filter, skip):
#             empty = not any(l in get_label_ids_by_category(crop, "present_annotated") for l in label.labelid)
#
#             gt_seg, mask = read_training(crop, label, gt_version)
#             binary_seg_gt = extract_binary_class(gt_seg, label)
#             offset, shape = get_offset_and_shape_from_training_crop(crop)
#             prediction = read_prediction(pred_path, label, offset, shape)
#             binary_seg_pred = threshold(prediction)
#             pix_count = np.sum(binary_seg_pred)
#             seg_empty = pix_count == 0
#             print("emtpy:", empty , "-", "seg_empty:", seg_empty)
#             if (not empty) or (not seg_empty):
#                 sitk_gt = sitk.GetImageFromArray(binary_seg_gt.astype(np.uint8))
#                 sitk_pred = sitk.GetImageFromArray(binary_seg_pred.astype(np.uint8))
#                 overlap_measures_filter.Execute(sitk_gt, sitk_pred)
#
#                 jaccard_index = overlap_measures_filter.GetJaccardCoefficient()
#                 jaccard_res[pred_path][crop["number"]] = jaccard_index
#
#                 dice_score = overlap_measures_filter.GetDiceCoefficient()
#                 dice_res[pred_path][crop["number"]] = dice_score
#
#                 fn = overlap_measures_filter.GetFalseNegativeError()
#                 fn_res[pred_path][crop["number"]] = fn
#
#                 fp = overlap_measures_filter.GetFalsePositiveError()
#                 fp_res[pred_path][crop["number"]] = fp
#
#             if seg_empty and empty:
#                 hd = 0
#             elif not seg_empty and not empty:
#                 hausdorff_distance_filter.Execute(sitk_gt, sitk_pred)
#                 hd = hausdorff_distance_filter.GetHausdorffDistance()
#             else:
#                 hd = np.Inf
#
#             ari = sklearn.metrics.adjusted_rand_score((binary_seg_gt).astype(np.uint8).flatten(),
#                                                               (binary_seg_pred).astype(np.uint8).flatten())
#             voi_split, voi_merge = cremi.evaluation.voi((binary_seg_pred).astype(np.uint8) + 1,
#                                                         (binary_seg_gt).astype(np.uint8) + 1,
#                                                         ignore_groundtruth = [])
#             voi = voi_split + voi_merge
#
#             hd_res[pred_path][crop["number"]] = hd
#             ari_res[pred_path][crop["number"]] = ari
#             voi_res[pred_path][crop["number"]] = voi
#             crop_sizes[crop["number"]] = np.prod(shape)
#
#         dice_res[pred_path]["average"] = score_average(dice_res[pred_path], crop_sizes)
#         jaccard_res[pred_path]["average"] = score_average(jaccard_res[pred_path], crop_sizes)
#         hd_res[pred_path]["average"] = score_average(hd_res[pred_path], crop_sizes)
#         fn_res[pred_path]["average"] = score_average(fn_res[pred_path], crop_sizes)
#         fp_res[pred_path]["average"] = score_average(fp_res[pred_path], crop_sizes)
#         ari_res[pred_path]["average"] = score_average(ari_res[pred_path], crop_sizes)
#         voi_res[pred_path]["average"] = score_average(voi_res[pred_path], crop_sizes)
#         n5file = zarr.open(pred_path, mode="a")
#         n5file[label.labelname].attrs["dice"] = dice_res[pred_path]
#         n5file[label.labelname].attrs["jaccard"] = jaccard_res[pred_path]
#         n5file[label.labelname].attrs["hausdorff"] = hd_res[pred_path]
#         n5file[label.labelname].attrs["fp"] = fp_res[pred_path]
#         n5file[label.labelname].attrs["fn"] = fn_res[pred_path]
#         n5file[label.labelname].attrs["ari"] = ari_res[pred_path]
#         #n5file[label.labelname].attrs["voi"] = voi_res[pred_path]
#
#     for sc_res, name in zip([dice_res, jaccard_res, fn_res, fp_res, ari_res, voi_res], ["dice", "jaccard", "fn", "fp",
#                                                                                   "ari", "voi"]):
#         for pred_path, res in sc_res.items():
#             print(pred_path, name, ":", res["average"])


def evaluate(label, predictions_to_compare, db_username, db_password, db_name="crops", gt_version="v0003",
             completion_min=5):
    # prepare db access
    client = pymongo.MongoClient("cosem.int.janelia.org:27017", username=db_username, password=db_password)
    db = client[db_name]  # db_name = "crops"
    collection = db[gt_version]  # gt_version = "v0003"
    skip = {"_id": 0, "number": 1, "labels": 1, "parent": 1, "dimensions": 1}

    # prepare dictionary to save results results[metric][pred_path][cropnumber]=score
    results = dict()
    for metric in EvaluationMetrics:
        print(metric.name)
        results[metric] = dict()

    for pred_path in predictions_to_compare:
        for metric in EvaluationMetrics:
            results[metric][pred_path] = dict()
        crop_sizes = dict()

        # get path for gt n5 dataset
        parent = get_parent(pred_path, label)
        filter = {"completion": {"$gte": completion_min}, "parent": parent}

        for crop in collection.find(filter, skip):
            # read and prepare segmentations
            gt_empty = not any(l in get_label_ids_by_category(crop, "present_annotated") for l in label.labelid)
            gt_seg, mask = read_training(crop, label, gt_version)
            binary_seg_gt = extract_binary_class(gt_seg, label)
            offset, shape = get_offset_and_shape_from_training_crop(crop)
            prediction = read_prediction(pred_path, label, offset, shape)
            binary_seg_pred = threshold(prediction)
            pix_count = np.sum(binary_seg_pred)
            seg_empty = pix_count == 0

            crop_sizes[crop["number"]] = np.prod(shape)

            # run evaluations
            evaluator = Evaluator(binary_seg_gt, binary_seg_pred, gt_empty=gt_empty, seg_empty=seg_empty)
            for metric in EvaluationMetrics:
                score = compute_score(metric, evaluator)
                if score is not None: # some scores cannot be computed on empty segmentations, return None
                    results[metric][pred_path][crop["number"]] = score

        for metric in EvaluationMetrics:
            results[metric][pred_path]["average"] = score_average(results[metric][pred_path], crop_sizes)

        # write results into attributes of dataset
        n5file = zarr.open(pred_path, mode="a")
        for metric in EvaluationMetrics:
            n5file[label.labelname].attrs[metric.name] = results[metric][pred_path]

    for metric in EvaluationMetrics:
        for pred_path, res in results[metric].items():
            print(pred_path, metric.name, ":", res["average"])

def main():
    label = Label("er", (16, 17, 18, 19, 20, 21, 22, 23))
    prediction_paths = "/nrs/cosem/cosem/training/v0003.2/setup27.1/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm_it{iteration:}.n5"
    predictions_to_compare = (prediction_paths.format(iteration=it) for it in range(25000, 500001, 25000))

    db_username = "root"
    db_password = "root"
    evaluate(label, predictions_to_compare, db_username, db_password)


if __name__ == "__main__":
    main()