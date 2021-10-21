from typing import Any, Dict, Optional, Tuple
from CNNectome.utils.setup_utils import autodiscover_label_to_crops, autodiscover_raw_datasets
import argparse
import collections
import itertools
import CNNectome.validation.organelles.segmentation_metrics as segmentation_metrics
import CNNectome.utils.cosem_db as cosem_db
import logging
import numpy as np

from CNNectome.validation.organelles.segmentation_metrics import sorting, EvaluationMetrics


def max_evaluated_iteration(query: Dict[str, Any],
                            db: cosem_db.MongoCosemDB) -> int:
    """
    Find maximum iteration that is found in the database for the given query.

    Args:
        query: Dictionary defining query for which to find max iteration.
        db: Database with evaluation results.

    Returns:
        Maximum iteration for `query`.
    """
    col = db.access("evaluation", db.training_version)
    max_it = col.aggregate([{"$match": query},
                            {"$sort": {"iteration": -1}},
                            {"$limit": 1},
                            {"$project": {"iteration": 1, "_id": 0}}])
    max_it = [m for m in max_it][0]
    return max_it["iteration"]


def convergence_iteration(query: Dict[str, Any],
                          db: cosem_db.MongoCosemDB,
                          check_evals_complete: bool = True) -> Tuple[int, int]:
    """
    Find the first iteration that meets the convergence criterion in 25k intervals. Convergence criterion is that both
    mean_false_distance and dice score indicate a decreasing performance for two consecutive evaluation points. If
    predictions don't produce above threshold segmentations by 500k iterations no higher iterations are considered.

    Args:
        query: Dictionary specifying which set of configuration to consider for the maximum iteration. This will
            typically contain keys for setups, label and crop.
        db: Database containing the evaluation results.
        check_evals_complete: Whether to first check whether the considered evaluations are consistent across the
            queries (i.e. same for all crops/labels/raw_datasets within one setup, at least to 500k, if above threshold
            by 500k at least to 700k). Should generally be set to True unless this has already been checked.

    Returns:
        The converged or maximum evaluated iteration and a flag indicating whether this represents a converged
        training. 0 for not converged training, 1 for converged trainings, 2 for trainings that have not reached above
        threshold predictions by 500k iterations, 3 for trainings that have not reached the convergence criterion but
        2,000,000 iterations.

    Raises:
        ValueError if no evaluations are found for given query.
    """
    query["iteration"] = {"$mod": [25000, 0]}
    metrics = ("dice", "mean_false_distance")
    col = db.access("evaluation", db.training_version)

    # check whether anything has been evaluated for this query type
    query_any = query.copy()
    query_any["metric"] = {"$in": metrics}
    if col.find_one(query_any) is None:
        raise ValueError("No evaluations found for query {0:}".format(query))
    if check_evals_complete:
        if not check_completeness(db, spec_query=query.copy()):
            return max_evaluated_iteration(query, db), 0
    if not above_threshold(db, query):
        return 500000, 2
    # get results and sort by iteration
    results = []
    for met in metrics:
        qy = query.copy()
        qy["metric"] = met
        qy["iteration"]["$lte"] = 2000000
        results.append(list(col.aggregate([{"$match": qy}, {"$sort": {"iteration": 1}}])))

    # check for convergence criterion
    for k in range(2, len(results[0])):
        this_one = [False, ] * len(metrics)
        for m_no, met in enumerate(metrics):
            if np.isnan(results[m_no][k]["value"]) and np.isnan(results[m_no][k-1]["value"]) and not np.isnan(
                    results[m_no][k-2]["value"]):
                this_one[m_no] = True
            else:
                if sorting(EvaluationMetrics[met]) == -1:
                    if results[m_no][k]["value"] <= results[m_no][k-1]["value"] < results[m_no][k-2]["value"]:
                        this_one[m_no] = True
                    elif (np.isnan(results[m_no][k]["value"]) and not np.isnan(results[m_no][k-1]["value"]) and not \
                            np.isnan(results[m_no][k-2]["value"]) and results[m_no][k-1]["value"] < results[m_no][
                        k-2]["value"]):
                        this_one[m_no] = True
                else:
                    if results[m_no][k]["value"] >= results[m_no][k-1]["value"] > results[m_no][k-2]["value"]:
                        this_one[m_no] = True
                    elif (np.isnan(results[m_no][k]["value"]) and not np.isnan(results[m_no][k-1]["value"]) and not \
                            np.isnan(results[m_no][k-2]["value"]) and results[m_no][k-1]["value"] > results[m_no][
                        k-2]["value"]):
                        this_one[m_no] = True
        if all(this_one):
            return results[0][k]["iteration"], 1
    if max_evaluated_iteration(query, db) >= 2000000:
        return 2000000, 3
    return results[0][-1]["iteration"], 0


def max_iteration_for_analysis(query: Dict[str, Any],
                               db: cosem_db.MongoCosemDB,
                               check_evals_complete: bool = False,
                               conv_it: Optional[Tuple[int, int]] = None) -> Tuple[int, bool]:
    """
    Find the first iteration that meets the convergence criterion like `convergence_iteration` but return a minimum
    iteration of 700k if the convergence criterion is met at a previous iteration. To avoid re-computation if
    `convergence_iteration` has explicitly been called before, the previous output can be passed in explicitly.

    Args:
        query: Dictionary specifying which set of configuration to consider for the maximum iteration. This will
            typically contain keys for setups, label and crop.
        db: Database containing the evaluation results.
        conv_it: Output of `convergence_iteration` if already known. Otherwise, None and `convergence_iteration` will
            be called.
        check_evals_complete: Whether to first check whether the considered evaluations are consistent across the
            queries (i.e. same for all crops/labels/raw_datasets within one setup, at least to 500k, if above threshold
            by 500k at least to 700k). Should generally be set to True unless this has already been checked.

    Returns:
        The max iteration. If none of the results produce above threshold segmentations False is returned. If the
        convergence condition isn't met anywhere or not evaluated to at least 700k iterations.

    Raises:
        ValueError if no evaluations are found for given query.
    """
    if conv_it is None:
        it, valid = convergence_iteration(query, db, check_evals_complete=check_evals_complete)
    else:
        it, valid = conv_it
    if valid != 2:
        it = max(it, 700000)
    return it, bool(valid)


def above_threshold(db: cosem_db.MongoCosemDB,
                    query: Dict[str, Any],
                    by: int = 500000) -> bool:
    """
    Check whether predictions for the given `query` are ever above threshold in the validation crop by iteration `by`.

    Args:
        db: Database with evaluation results.
        query: Dictionary defining query for which to check for whether they're above threshold.
        by: Only check evaluation results up to this iteration (inclusive).

    Returns:
        True if there are any results above threshold by iteration `by`. False otherwise.
    """
    qy = query.copy()
    qy["metric"] = "mean_false_distance"
    qy["value"] = {"$gt": 0}
    qy["iteration"] = {"$mod": [25000, 0], "$lte": by}
    eval_col = db.access("evaluation", db.training_version)
    return not(eval_col.find_one(qy) is None)


def check_completeness(db: cosem_db.MongoCosemDB,
                       setup: Optional[str] = None,
                       metric_params: Optional[Dict[str, int]] = None,
                       threshold: int = None,
                       spec_query: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check whether for the given configuration each relevant label/raw_dataset is evaluated for all metrics for the same
    iterations (in 25k intervals, starting from 25k).

    Args:
        db: Database with crop information and evaluation results.
        setup: Network setup to check.
        metric_params: Dictionary with metric parameters.
        threshold: Value at which predictions were thresholded for evaluation.
        spec_query: Alternatively to specifying these arguments they can be fed in as a dictionary.

    Returns:
        False if any evaluations are missing/inconsistent. Otherwise, True.
    """
    if spec_query is None:
        spec_query = dict()
    if "refined" in spec_query:
        if spec_query["refined"]:
            logging.info("Check for refined predictions not necessary")
            return True
    if setup is None:
        try:
            setup = spec_query["setup"]
        except KeyError:
            raise ValueError("setup needs to be specified as kwarg or in spec_query")

    if threshold is None:
        try:
            threshold = spec_query["threshold"]
        except KeyError:
            raise ValueError("threshold needs to be specified as kwarg or in spec_query")
    if metric_params is None:
        try:
            metric_params = spec_query["metric_params"]
        except KeyError:
            logging.warning("metric_params not specified as kwarg or spec_query, defaulting to tol_distance=40,"
                            "clip_distance=200")
            metric_params = {"clip_distance": 200, "tol_distance": 40}

    if "raw_dataset" in spec_query:
        if isinstance(spec_query["raw_dataset"], dict):
            try:
                raw_datasets = spec_query["raw_dataset"]["$in"]
            except KeyError:
                raise NotImplementedError(
                    "don't know how to do check with query {0:} for raw_dataset".format(spec_query["raw_dataset"]))
        else:
            raw_datasets = [spec_query["raw_dataset"], ]
    else:
        raw_datasets = autodiscover_raw_datasets(setup)

    if "crop" in spec_query:
        if not (isinstance(spec_query["crop"], int) or isinstance(spec_query["crop"], str)):
            raise NotImplementedError("can't check query with complicated query for crop")
        else:
            spec_query["crop"] = str(spec_query["crop"])
    if "label" in spec_query and "crop" in spec_query:
        label_to_cropnos = {spec_query["label"]: [spec_query["crop"],]}
    else:
        label_to_cropnos = autodiscover_label_to_crops(setup, db)
        if "label" in spec_query:
            label_to_cropnos = dict((k, v) for k, v in label_to_cropnos.items() if k in spec_query["label"])
        if "crop" in spec_query:
            for k, v in label_to_cropnos:
                new_v = [vv for vv in v if vv in [spec_query["crop"]]]
                if len(new_v) > 0:
                    label_to_cropnos[k] = new_v
                else:
                    del label_to_cropnos[k]
    if len(label_to_cropnos) == 0:
        return True

    eval_col = db.access("evaluation", db.training_version)
    will_return = True
    iterations_col = []
    for met in segmentation_metrics.EvaluationMetrics:
        met_specific_params_nested = segmentation_metrics.filter_params(metric_params, met)
        if met_specific_params_nested:
            for k, v in met_specific_params_nested.items():
                if not isinstance(v, collections.Iterable):
                    met_specific_params_nested[k] = [v]
            met_specific_params_it = list(itertools.chain(*[[{k:vv} for vv in v] for k, v in met_specific_params_nested.items()]))
        else:
            met_specific_params_it = [dict(), ]
        for met_specific_params in met_specific_params_it:
            for lblname, cropnos in label_to_cropnos.items():
                for cropno in cropnos:
                    for raw_ds in raw_datasets:
                        query = {"setup": setup,
                                 "raw_dataset": raw_ds,
                                 "crop": cropno,
                                 "label": lblname,
                                 "threshold": threshold,
                                 "refined": False,
                                 "iteration": {"$mod": [25000, 0]},
                                 "metric": met,
                                 "metric_params": met_specific_params
                                 }
                        iterations = list(eval_col.aggregate([{"$match": query},
                                                              {"$sort": {"iteration": 1}},
                                                              {"$project": {"iteration": True, "_id": False}}]))
                        iterations_col.append([it["iteration"] for it in iterations])
                        if len(iterations_col) > 1:
                            if iterations_col[-1] != iterations_col[-2]:
                                print("Results for query {0:} not matching: {1:}".format(query,
                                                                                         iterations_col[-1]))
                                will_return = False
    if not iterations_col[-1] == list(range(25000, iterations_col[-1][-1]+1, 25000)):
        print("Missing checkpoints, found: {0:}".format(iterations_col[-1]))
        will_return = False
    if not iterations_col[-1][-1] >= 500000:
        print("Not evaluated to 500000 iterations.")
        will_return = False

    # check till 700k if pos results until 500k
    if will_return:
        for lblname, cropnos in label_to_cropnos.items():
            for cropno in cropnos:
                for raw_ds in raw_datasets:
                    query = {"setup": setup,
                             "raw_dataset": raw_ds,
                             "crop": cropno,
                             "label": lblname,
                             "threshold": threshold,
                             "refined": False}

                    # if above threshold results are found by 500k iterations, network should be evaluated until at
                    # least 700k iterations
                    if above_threshold(db, query):
                        query["iteration"] = 700000  # if 700k exists the ones in between exist (checked abvoe)
                        if eval_col.find_one(query) is None:
                            print("For query {0:}, above threshold results are found by 500k iterations but network "
                                  "isn't evaluated to at least 700k".format(query))
                            will_return = False
    return will_return


def check_convergence(setup: str,
                      threshold: int,
                      db: cosem_db.MongoCosemDB,
                      tol_distance: int = 40,
                      clip_distance: int = 200) -> bool:
    """
    Check whether the given setup training has reached our convergence criterion.

    Args:
        setup: Network setup to check.
        threshold:  Value at which predictions were thresholded for evaluation.
        db: Database with crop information and evaluation results.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using a metric with clip distance, otherwise not used.

    Returns:
        True if training is considered done, otherwise False
    """
    metric_params = {"tol_distance": tol_distance,
                     "clip_distance": clip_distance}
    if not check_completeness(db, setup, metric_params, threshold):
        return False
    will_return = True
    label_to_cropnos = autodiscover_label_to_crops(setup, db)
    raw_datasets = autodiscover_raw_datasets(setup)
    for lblname, cropnos in label_to_cropnos.items():
        for cropno in cropnos:
            for raw_ds in raw_datasets:
                query = {"setup": setup,
                         "raw_dataset": raw_ds,
                         "crop": cropno,
                         "label": lblname,
                         "threshold": threshold,
                         "refined": False}
                max_it, valid = max_iteration_for_analysis(query, db, check_evals_complete=False)
                if valid is False:
                    will_return = False
                    print("Results for query {0:} show not converged training: {1:}".format(query, max_it))
                else:
                    print("Success for query {0:}: {1:}".format(query, max_it))
    return will_return


def main() -> None:
    parser = argparse.ArgumentParser("Check whether trainings are complete and converged.")
    parser.add_argument("type", type=str, choices=["completeness", "convergence"],
                        help="Pick whether to check convergence or just completeness.")
    parser.add_argument("setup", type=str, help="Network setup to check.")
    parser.add_argument("--tol_distance", type=int, default=40, nargs="+",
                        help="Tolerance distance to check for with metrics using tolerance distance.")
    parser.add_argument("--clip_distance", type=int, default=200, nargs="+",
                        help="Clip distance to check for which metrics using clip distance.")
    parser.add_argument("--threshold", type=int, default=127,
                        help="Threshold to have been applied on top of raw predictions.")
    parser.add_argument("--training_version", type=str, default="v0003.2", help="Version of training")
    parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth")
    parser.add_argument("--check_private_db", action="store_true")
    args = parser.parse_args()
    db = cosem_db.MongoCosemDB(training_version=args.training_version, gt_version=args.gt_version, 
        write_access=args.check_private_db)
    metric_params = {"tol_distance": args.tol_distance,
                     "clip_distance": args.clip_distance}
    if args.type == "completeness":
        print(check_completeness(db, args.setup, metric_params, threshold=args.threshold))
    else:
        print(check_convergence(args.setup, args.threshold, db))


if __name__ == "__main__":
    main()
