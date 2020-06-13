from CNNectome.validation.organelles.segmentation_metrics import *
from CNNectome.validation.organelles.run_evaluation import construct_pred_path, autodetect_labelnames, autodetect_iteration, get_all_annotated_labelnames
from CNNectome.utils import cosem_db
from CNNectome.utils.hierarchy import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
from more_itertools import always_iterable, repeat_last

import warnings
import os

db_host = "cosem.int.janelia.org:27017"
gt_version = "v0003"
training_version = "v0003.2"


def y_scale_group(metric):
    switcher = {
        EvaluationMetrics.dice:                                 0,
        EvaluationMetrics.jaccard:                              0,
        EvaluationMetrics.hausdorff:                            2,
        EvaluationMetrics.false_negative_rate:                  0,
        EvaluationMetrics.false_negative_rate_with_tolerance:   0,
        EvaluationMetrics.false_positive_rate:                  0,
        EvaluationMetrics.false_discovery_rate:                 0,
        EvaluationMetrics.false_positive_rate_with_tolerance:   0,
        EvaluationMetrics.voi:                                  0,
        EvaluationMetrics.mean_false_distance:                  1,
        EvaluationMetrics.mean_false_positive_distance:         1,
        EvaluationMetrics.mean_false_negative_distance:         1,
        EvaluationMetrics.mean_false_distance_clipped:          1,
        EvaluationMetrics.mean_false_negative_distance_clipped: 1,
        EvaluationMetrics.mean_false_positive_distance_clipped: 1,
        EvaluationMetrics.precision_with_tolerance:             0,
        EvaluationMetrics.recall_with_tolerance:                0,
        EvaluationMetrics.f1_score_with_tolerance:              0,
        EvaluationMetrics.precision:                            0,
        EvaluationMetrics.recall:                               0,
        EvaluationMetrics.f1_score:                             0,
    }
    return switcher.get(metric)


def plot_validation(x_axis, validations, metrics, metric_params, db):
    axs = [None, ] * 3
    ref_ax = y_scale_group(metrics[0])
    fig, axs[ref_ax] = plt.subplots()
    x_ticks = set()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, metric in enumerate(metrics):
        x_axis_values = []
        y_axis_values = []
        for val in validations:
            specific_metric_params = filter_params(metric_params, metric)
            eval_res = db.read_evaluation_result(val["path"], val["dataset"], val["setup"], val["iteration"],
                                                 val["label"].labelname, val["crop"]["number"], val["threshold"], metric,
                                                 specific_metric_params)
            if eval_res is None:
                raise ValueError("Did not find {0:} in database".format((val["path"], val["dataset"], val["setup"], val["iteration"],
                                                 val["label"].labelname, val["crop"]["number"], val["threshold"], metric,
                                                 specific_metric_params)))
            eval_res.update(specific_metric_params)
            x_axis_values.append(eval_res[x_axis])
            y_axis_values.append(eval_res["value"])
        ax = axs[y_scale_group(metric)]
        if ax is None:
            axs[y_scale_group(metric)] = axs[ref_ax].twinx()
            ax = axs[y_scale_group(metric)]
        line, = ax.plot(x_axis_values, y_axis_values, label=metric, c = cycle[i])
        opt = best(metric)(y_axis_values)
        ax.plot(x_axis_values[opt], y_axis_values[opt], c = line.get_color(), alpha = 0.5, marker = 'o')
        x_ticks.union(set(x_axis_values))
    for ax in axs:
        if ax is not None:
            ax.legend()
    axs[ref_ax].xticks = sorted(list(x_ticks))
    plt.show()


def main():
    parser = argparse.ArgumentParser("Plot evaluation results")
    parser.add_argument('x_axis', type=str, help="quantitity to plot on x-axis")
    parser.add_argument("--setup", type=str, nargs='+', default=None,
                        help="network setup from which to evaluate a prediction, e.g. setup01")
    parser.add_argument("--iteration", type=int, nargs='+', default=None,
                        help="network iteration from which to evaluate prediction, e.g. 725000")
    parser.add_argument("--label", type=str, nargs='+', default=None,
                        help="label for which to evaluate prediction, choices: " + ", ".join(list(hierarchy.keys())))
    parser.add_argument("--crop", type=int, nargs='+', default=None,
                        help="number of crop with annotated groundtruth, e.g. 110")
    parser.add_argument("--threshold", type=int, default=128, nargs='+',
                        help="threshold to apply on distances")
    parser.add_argument("--pred_path", type=str, default=None, nargs='+',
                        help="path of n5 file containing predictions")
    parser.add_argument("--pred_ds", type=str, default=None, nargs='+',
                        help="dataset of the n5 file containing predictions")
    parser.add_argument("--metric", type=str, default=None, help="metric to evaluate", nargs='+',
                        choices=list(em.value for em in EvaluationMetrics))
    parser.add_argument("--clip_distance", type=int, default=200,
                        help="Parameter used for clipped false distances. False distances larger than the value of "
                             "this parameter are reduced to this value.")
    parser.add_argument("--tol_distance", type=int, default=40,
                        help="Parameter used for counting false negatives/positives with a tolerance. Only false "
                             "predictions that are farther than this value from the closest pixel where they would be "
                             "correct are counted.")
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")

    args = parser.parse_args()
    db = cosem_db.MongoCosemDB(args.db_username, args.db_password, host=db_host, gt_version=gt_version,
                               training_version=training_version)
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
            pred_path = construct_pred_path(setup, iteration, crop)
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
                if ll not in get_all_annotated_labelnames(crop):
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
                                "You specified pred_path as well as iteration. The iteration does not match the iteration in the attributes of the prediction."
                            )
                else:
                    iteration = autodetect_iteration(pred_path, ds)
                    if iteration is None:
                        raise ValueError(
                            "Please sepcify iteration, it is not contained in the prediction metadata."
                        )

                if pred_path != construct_pred_path(setup, iteration, crop):
                    warnings.warn(
                            "You specified pred_path as well as setup and the pred_path does not match the standard location."
                        )
            if not os.path.exists(os.path.join(pred_path, ds)):
                raise ValueError('{0:} not found'.format(os.path.join(pred_path, ds)))
            validations.append({
                'path': pred_path,
                "dataset": ds,
                "setup": setup,
                "iteration": iteration,
                "label": hierarchy[ll],
                "crop": crop,
                "threshold": thr
            })
    plot_validation(args.x_axis, validations, metric, metric_params, db)


if __name__ == "__main__":
    main()
