from CNNectome.utils import crop_utils
from CNNectome.validation.organelles.segmentation_metrics import *
from CNNectome.utils import cosem_db
from CNNectome.utils.hierarchy import hierarchy
from CNNectome.utils.label import Label
import os
import numpy as np
import zarr
import argparse
from more_itertools import repeat_last, always_iterable
import tabulate
import warnings
import itertools
import scipy.ndimage

db_host = "cosem.int.janelia.org:27017"
gt_version = "v0003"
training_version = "v0003.2"
eval_results_csv_folder = "/groups/cosem/cosem/computational_evaluation/{training_version:}/csv/".format(training_version=training_version)


def downsample(arr, factor=2):
    return arr[(slice(None, None, factor),) * arr.ndim]


def get_parent(prediction_path, label):
    n5file = zarr.open(prediction_path, mode="r")
    pred = n5file[label.labelname]
    return pred.attrs["raw_data_path"]


def add_constant(seg, resolution, label):
    inner_distance = scipy.ndimage.distance_transform_edt(scipy.ndimage.morphology.binary_erosion(seg, border_value=1,
                                                                                                  structure=scipy.ndimage.generate_binary_structure(seg.ndim, seg.ndim)), sampling=resolution)
    outer_distance = scipy.ndimage.distance_transform_edt(np.logical_not(seg), sampling=resolution)
    signed_distance = inner_distance - outer_distance + label.add_constant
    gt_seg = apply_threshold(signed_distance, thr=0)
    return gt_seg


def read_gt(crop, label, resolution, gt_version="v0003"):
    n5file = zarr.open(crop["parent"], mode="r")
    blueprint_label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/{{label:}}"
    label_ds = blueprint_label_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    if label.separate_labelset:
        label_ds_name = label.labelname
    else:
        label_ds_name = "all"
    gt_seg = n5file[label_ds.format(label=label_ds_name)]
    if label.add_constant is not None and label.add_constant != 0:
        gt_seg = add_constant(gt_seg, resolution, label)
    gt_seg = downsample(gt_seg)

    blueprint_labelmask_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/masks/{{label:}}"
    labelmask_ds = blueprint_labelmask_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    labelmask_ds.format(label=label_ds_name)
    if labelmask_ds in n5file:
        mask = n5file[labelmask_ds]
    else:
        if label.generic_label is not None:
            specific_labels = list(set(label.labelid) - set(label.generic_label))
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


def apply_threshold(prediction, thr=127):
    return (prediction >= thr).astype(np.bool)


def read_prediction(prediction_path, pred_ds, offset, shape):
    n5file = zarr.open(prediction_path, mode="r")
    pred = n5file[pred_ds]
    resolution = pred.attrs["resolution"][::-1]
    sl = tuple(slice(int(o), int(o+s), None) for o, s in zip(offset, shape))

    return pred[sl], resolution


def pred_path_without_iteration(setup, crop, s1):
    default_pred_path = "/nrs/cosem/cosem/training/{0:}/".format(training_version)
    setup_path = os.path.join(default_pred_path, setup)
    pred_path = os.path.join(setup_path, crop_utils.get_data_path(crop, s1))
    return pred_path.split("it{0:}.n5")[0]+"it"


def construct_pred_path(setup, iteration, crop, s1):
    default_pred_path = '/nrs/cosem/cosem/training/v0003.2/'
    setup_path = os.path.join(default_pred_path, setup)
    pred_path = os.path.join(setup_path, crop_utils.get_data_path(crop, s1).format(iteration))

    return pred_path


def autodetect_labelnames(path, crop):
    n5 = zarr.open(zarr.N5Store(path), 'r')
    labels_predicted = set(n5.array_keys())
    labels_in_crop = set(crop_utils.get_all_annotated_labelnames(crop))
    labels_eval = list(labels_predicted.intersection(labels_in_crop))
    return labels_eval


def autodetect_iteration(path, ds):
    n5_ds = zarr.open(zarr.N5Store(path), 'r')[ds]
    try:
        return n5_ds.attrs['iteration']
    except KeyError:
        return None


def run_validation(pred_path, pred_ds, setup, iteration, label, crop, threshold, metrics, metric_params, db=None,
                   csvh=None, save=False, overwrite=False):
    results = dict()
    n5 = zarr.open(pred_path, mode="r")
    raw_dataset = n5[pred_ds].attrs["raw_ds"]
    parent_path = n5[pred_ds].attrs["raw_data_path"]
    for metric in metrics:
        metric_specific_params = filter_params(metric_params, metric)
        # check db
        if save:
            query = {"path": pred_path, "dataset": pred_ds, "setup": setup, "iteration": iteration,
                     "label": label.labelname, "crop": crop["number"], "threshold": threshold, "metric": metric,
                     "metric_params": metric_specific_params, "raw_dataset": raw_dataset, "parent_path": parent_path}
            db_entry = db.read_evaluation_result(query)
            if db_entry is not None:
                if overwrite:
                    db.delete_evaluation_result(query)
                    csvh.delete_evaluation_result(query)
                else:
                    results[metric] = db_entry['value']
                    print('.', end='', flush=True)

    if set(results.keys()) != set(metrics):
        gt_empty = not any(l in crop_utils.get_label_ids_by_category(crop, "present_annotated") for l in label.labelid)
        offset, shape = crop_utils.get_offset_and_shape_from_crop(crop)
        prediction, resolution = read_prediction(pred_path, pred_ds, offset, shape)
        gt_seg, mask = read_gt(crop, label, resolution, gt_version)
        gt_binary = extract_binary_class(gt_seg, label)
        test_binary = apply_threshold(prediction, thr=threshold)
        pred_empty = np.sum(test_binary) == 0
        evaluator = Evaluator(gt_binary, test_binary, gt_empty, pred_empty, metric_params, resolution)
        remaining_metrics = list(set(metrics) - set(results.keys()))
        for metric in remaining_metrics:
            metric_specific_params = filter_params(metric_params, metric)
            score = evaluator.compute_score(EvaluationMetrics[metric])

            if save:
                document = {"path": pred_path, "dataset": pred_ds, "setup": setup, "iteration": iteration,
                            "label": label.labelname, "crop": crop["number"], "threshold": threshold, "metric": metric,
                            "metric_params": metric_specific_params, "value": score, "raw_dataset": raw_dataset, "parent_path":
                            parent_path}
                db.write_evaluation_result(document)
                csvh.write_evaluation_result(document)

            results[metric] = score
            print('.', end='', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser("Evaluate predictions")
    parser.add_argument("setup", type=str, nargs='+', default=None,
                        help="network setup from which to evaluate a prediction, e.g. setup01")
    parser.add_argument("--iteration", type=int, nargs='+', default=None,
                        help="network iteration from which to evaluate prediction, e.g. 725000")
    parser.add_argument("--label", type=str, nargs='+', default=None,
                        help="label for which to evaluate prediction, choices: " + ", ".join(list(hierarchy.keys())))
    parser.add_argument("--crop", type=int, nargs='+', default=None,
                        help="number of crop with annotated groundtruth, e.g. 110")
    parser.add_argument("--threshold", type=int, default=127, nargs='+',
                        help="threshold to apply on distances")
    parser.add_argument("--pred_path", type=str, default=None, nargs='+',
                        help="path of n5 file containing predictions")
    parser.add_argument("--pred_ds", type=str, default=None, nargs='+',
                        help="dataset of the n5 file containing predictions")
    parser.add_argument("--metric", type=str, default=None, help="metric to evaluate",
                        choices=list(em.value for em in EvaluationMetrics), nargs="+")
    parser.add_argument("--clip_distance", type=int, default=200,
                        help="Parameter used for clipped false distances. False distances larger than the value of "
                             "this parameter are reduced to this value.")
    parser.add_argument("--tol_distance", type=int, default=40,
                        help="Parameter used for counting false negatives/positives with a tolerance. Only false "
                             "predictions that are farther than this value from the closest pixel where they would be "
                             "correct are counted.")
    parser.add_argument("--save", action='store_true',
                        help="save to database and csv file")
    parser.add_argument("--overwrite", action='store_true',
                        help="overwrite existing entries in database and csv")
    parser.add_argument("--s1", action='store_true', help="use s1 standard directory")
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--dry-run", action='store_true',
                        help="show list of evaluations that would be run with given arguments without compute anything")

    args = parser.parse_args()
    db = cosem_db.MongoCosemDB(args.db_username, args.db_password, host=db_host, gt_version=gt_version,
                               training_version=training_version)
    csvhandler = cosem_db.CosemCSV(eval_results_csv_folder)
    if args.overwrite and not args.save:
        raise ValueError("Overwriting should only be set if save is also set")
    if args.crop is None:
        crops = db.get_all_validation_crops()
    else:
        crops = []
        for cno in args.crop:
            c = db.get_crop_by_number(cno)
            if c is None:
                raise ValueError("Did not find crop {0:} in database".format(cno))
            crops.append(c)
    if args.metric is None:
        metric = list(em.value for em in EvaluationMetrics)
    else:
        metric = list(always_iterable(args.metric))

    metric_params = dict()
    metric_params['clip_distance'] = args.clip_distance
    metric_params['tol_distance'] = args.tol_distance

    num_validations = max(len(crops), len(list(always_iterable(args.setup))), len(list(always_iterable(args.iteration))),
                          len(list(always_iterable(args.label))), len(list(always_iterable(args.pred_path))),
                          len(list(always_iterable(args.pred_ds))), len(list(always_iterable(args.threshold))))
    print("\nWill run the following validations:\n")
    validations = []
    for valno, crop, setup, iteration, label, pred_path, pred_ds, thr in zip(range(num_validations),
                                                                             repeat_last(always_iterable(crops)),
                                                                             repeat_last(always_iterable(args.setup)),
                                                                             repeat_last(
                                                                                 always_iterable(args.iteration)),
                                                                             repeat_last(always_iterable(args.label)),
                                                                             repeat_last(
                                                                                 always_iterable(args.pred_path)),
                                                                             repeat_last(always_iterable(args.pred_ds)),
                                                                             repeat_last(
                                                                                 always_iterable(args.threshold))):

        if pred_ds is not None:
            if label is None:
                raise ValueError("If pred_ds is specified, label can't be autodetected")

        if pred_path is None:
            if iteration is None:
                raise ValueError("Either pred_path or iteration must be specified")
            pred_path = construct_pred_path(setup, iteration, crop, args.s1)
        if not os.path.exists(pred_path):
            raise ValueError("{0:} not found".format(pred_path))
        if not os.path.exists(os.path.join(pred_path, 'attributes.json')):
            raise RuntimeError("N5 is incompatible with zarr due to missing attributes files. Consider running"
                               " `add_missing_n5_attributes {0:}`".format(pred_path))
        if label is None:
            labels = autodetect_labelnames(pred_path, crop)
        else:
            labels = list(always_iterable(label))
            for ll in labels:
                if ll not in crop_utils.get_all_annotated_labelnames(crop):
                    raise ValueError("Label {0:} not annotated in crop {1:}".format(ll, crop['number']))

        for ll in labels:
            if pred_ds is None:
                ds = ll
            else:
                ds = pred_ds
            if pred_path is not None:  # and (setup is not None or iteration is not None):
                if not os.path.exists(os.path.join(pred_path, ds)):
                    raise ValueError('{0:} not found'.format(os.path.join(pred_path, ds)))
                if iteration is not None:
                    auto_it = autodetect_iteration(pred_path, ds)
                    if auto_it is not None:
                        if iteration != autodetect_iteration(pred_path, ds):
                            raise ValueError(
                                "You specified pred_path as well as iteration. The iteration does not match the "
                                "iteration in the attributes of the prediction."
                            )
                else:
                    iteration = autodetect_iteration(pred_path, ds)
                    if iteration is None:
                        raise ValueError(
                            "Please sepcify iteration, it is not contained in the prediction metadata."
                        )

                if pred_path != construct_pred_path(setup, iteration, crop, args.s1):
                    warnings.warn(
                            "You specified pred_path as well as setup and the pred_path does not match the standard "
                            "location."
                        )
            if not os.path.exists(os.path.join(pred_path, ds)):
                raise ValueError('{0:} not found'.format(os.path.join(pred_path, ds)))
            n5 = zarr.open(pred_path, mode="r")
            raw_ds = n5[ds].attrs["raw_ds"]
            parent_path = n5[ds].attrs["raw_data_path"]
            validations.append([pred_path, ds, setup, iteration, hierarchy[ll], crop, raw_ds, parent_path, thr])

    tabs = [(pp, d, s, i, ll.labelname, c['number'], r_ds, parent, t, m, filter_params(metric_params, m)) for
            (pp, d, s, i, ll, c, r_ds, parent, t), m in itertools.product(validations, metric)]
    print(tabulate.tabulate(tabs, ["Path", "Dataset", "Setup", "Iteration", "Label", "Crop", "Raw Dataset",
                                   "Parent Path", "Threshold", "Metric", "Metric Params"]))

    if not args.dry_run:
        print("\nRunning Evaluations:")
        for val_params in validations:
            pp, d, s, i, ll, c , r_ds, parent, t = val_params
            results = run_validation(pp, d, s, i, ll, c, t, metric, metric_params, db, csvhandler, args.save,
                                     args.overwrite)
            val_params.append(results)
        print("\nResults Summary:")
        tabs = [(pp, d, s, i, ll.labelname, c['number'], r_ds, parent, t, m, filter_params(metric_params, m), v) for
                (pp, d, s, i, ll, c, r_ds, parent, t, r) in validations for m, v in r.items()]
        print(tabulate.tabulate(tabs, ["Path", "Dataset", "Setup", "Iteration", "Label", "Crop",
                                       "Raw Dataset", "Parent Path", "Threshold", "Metric",
                                       "Metric Params", "Value"]))


if __name__ == "__main__":
    main()
