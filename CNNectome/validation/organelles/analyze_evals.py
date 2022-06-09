import csv
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from CNNectome.utils import config_loader, cosem_db
from CNNectome.utils.crop_utils import (check_label_in_crop,
                                        get_label_ids_by_category)
from CNNectome.utils.hierarchy import hierarchy
from CNNectome.utils.setup_utils import autodiscover_labels
from CNNectome.validation.organelles.check_consistency import \
    max_iteration_for_analysis
from CNNectome.validation.organelles.run_evaluation import *


def _best_automatic(db: cosem_db.MongoCosemDB,
                    label: str,
                    setups: Sequence[str],
                    cropno: Union[Sequence[str], Sequence[int]],
                    metric: str,
                    raw_ds: Optional[Sequence[str]] = None,
                    tol_distance: int = 40,
                    clip_distance: int = 200,
                    threshold: int = 127,
                    test: bool = False) -> Dict[str, Any]:
    metric_params = dict()
    metric_params["clip_distance"] = clip_distance
    metric_params["tol_distance"] = tol_distance
    filtered_params = filter_params(metric_params, metric)

    setups = [setup for setup in setups if label in [lbl.labelname for lbl in autodiscover_labels(setup)]]
    # in test mode the remaining validation crops are used for determining best configuration
    if test:
        cropnos_query = [crop["number"] for crop in db.get_all_validation_crops()]
        for cno in cropno:
            cropnos_query.pop(cropnos_query.index(str(cno)))
        cropnos_query = [cno for cno in cropnos_query if check_label_in_crop(hierarchy[label], db.get_crop_by_number(cno))]
    else:
        cropnos_query = cropno
    if len(cropnos_query) == 0:  # if no crops remain return without result
        final = {"value": None, "iteration": None, "label": label, "metric": metric, "metric_params": filtered_params,
                 "refined": False, "threshold": threshold, "setup": setups[0] if len(setups) == 1 else None,
                 "crop": cropno[0] if len(cropno) == 1 else {"$in": cropno}}
        if raw_ds is not None:
            final["raw_dataset"] = raw_ds[0] if len(raw_ds) == 1 else {"$in": raw_ds}
        return final

    # find max iterations and put corresponding conditions in query
    conditions = []
    for setup in setups:  # several setups if both iteration and setup are being optimized ("across-setups")

        max_its = []

        for cno in cropnos_query:
            maxit_query = {"label": label,
                           "crop": str(cno),
                           "threshold": threshold,
                           "refined": False,
                           "setup": setup}
            if raw_ds is not None:
                maxit_query["raw_dataset"] = {"$in": raw_ds}
            maxit, valid = max_iteration_for_analysis(maxit_query, db)
            max_its.append(maxit)

        conditions.append({"setup": setup,
                           "iteration": {"$lte": max(max_its)}})

    if len(conditions) > 1:
        match_query = {"$or": conditions}
    else:
        match_query = conditions[0]

    # prepare aggregation of best configuration on the database
    aggregator = []

    # match
    match_query.update({"crop": {"$in": cropnos_query},
                        "label": label,
                        "metric": metric,
                        "metric_params": filtered_params,
                        "threshold": threshold,
                        "value": {"$ne": np.nan},
                        "refined": False})
    if raw_ds is not None:
        match_query["raw_dataset"] = {"$in": raw_ds}
    aggregator.append({"$match": match_query})

    # for each combination of setup and iteration, and raw_dataset if relevant, average across the matched results
    crossval_group = {"_id": {"setup": "$setup",
                              "iteration": "$iteration"},
                      "score": {"$avg": "$value"}}
    if raw_ds is not None:
        crossval_group["_id"]["raw_dataset"] = "$raw_dataset"
    aggregator.append({"$group": crossval_group})

    # sort (descending/ascending determined by metric) by averaged score
    aggregator.append({"$sort": {"score": sorting(metric), "_id.iteration": 1}})

    # only need max so limit results to one (mongodb can take advantage of this for sort)
    aggregator.append({"$limit": 1})

    # extract setup and iteration, and raw_dataset if relevant, in the end
    projection = {"setup": "$_id.setup",
                  "iteration": "$_id.iteration",
                  "_id": 0}
    if raw_ds is not None:
        projection["raw_dataset"] = "$_id.raw_dataset"
    aggregator.append({"$project": projection})

    # run the aggregation on the evaluation database
    col = db.access("evaluation", (db.training_version, db.gt_version))
    best_config = list(col.aggregate(aggregator))

    if len(best_config) == 0:  # if no results are found, return at this point
        final = match_query.copy()
        # final result should have actual cropno
        if len(cropno) == 1:
            final["crop"] = cropno[0]
        else:
            final["crop"] = {"$in": cropno}
        final.update({"setup": None, "value": None, "iteration": None})
        return final
    else:
        best_config = best_config[0]

    all_best = []
    for cno in cropno:
        query_best = {"label": label,
                      "crop": str(cno),
                      "metric": metric,
                      "setup": best_config["setup"],
                      "metric_params": filtered_params,
                      "threshold": threshold,
                      "iteration": best_config["iteration"],
                      "refined": False}
        if raw_ds is not None:
            query_best["raw_dataset"] = best_config["raw_dataset"]
        best_this = db.find(query_best)
        if len(best_this) != 1:
            print("query:", query_best)
            print("results:", list(best_this))
        assert len(best_this) == 1, "Got more than one result for best"
        all_best.append(best_this[0])

    # average results for the case of several crops
    final = dict()
    final["value"] = np.mean([ab["value"] for ab in all_best])

    # assemble all entries that are shared by the best result for each crop
    all_keys = set(all_best[0].keys()).intersection(*(d.keys() for d in all_best)) - {"value"}
    for k in all_keys:
        if all([ab[k] == all_best[0][k] for ab in all_best]):
            final[k] = all_best[0][k]
    return final


def _best_manual(db: cosem_db.MongoCosemDB,
                 label: str,
                 setups: Sequence[str],
                 cropno: Union[int, str],
                 raw_ds: Optional[Sequence[str]] = None) -> Optional[
      Dict[str, Union[str, int, bool]]]:

    # read csv file containing results of manual evaluation, first for best iteration
    c = db.get_crop_by_number(str(cropno))
    csv_folder_manual = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], db.training_version,
                                     "manual")
    csv_file_iterations = open(os.path.join(csv_folder_manual, c["dataset_id"] + "_iteration.csv"), "r")
    fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
    reader = csv.DictReader(csv_file_iterations, fieldnames)

    # look for all possible matches with the given query
    best_manuals = []
    for row in reader:
        if row["labelname"] == label and row["setup"] in setups:
            if raw_ds is None or row["raw_dataset"] in raw_ds:
                manual_result = {"setup": row["setup"],
                                 "label": row["labelname"],
                                 "iteration": int(row["iteration"]),
                                 "raw_dataset": row["raw_dataset"],
                                 "crop": str(cropno),
                                 "metric": "manual"}
                best_manuals.append(manual_result)
    if len(best_manuals) == 0:  # no manual evaluations with the given constraints were done
        return None
    elif len(best_manuals) == 1:  # if there's only one match it has to be the best one
        return best_manuals[0]
    else:  # if there's several matches check the setup results for overall best
        # read csv file containing results of manual evaluations, now for best setup per label/crop
        csv_file_setups = open(os.path.join(csv_folder_manual, c["dataset_id"] + "_setup.csv"), "r")
        reader = csv.DictReader(csv_file_setups, fieldnames)
        for row in reader:
            if row["labelname"] == label and row["setup"] in setups:
                if raw_ds is None or row["raw_dataset"] in raw_ds:
                    manual_result_best = {"setup": row["setup"],
                                          "label": row["labelname"],
                                          "iteration": int(row["iteration"]),
                                          "raw_dataset": row["raw_dataset"],
                                          "crop": str(cropno),
                                          "metric": "manual",
                                          "refined": False}
                    return manual_result_best
    return None


def best_result(db: cosem_db.MongoCosemDB,
                label: str,
                setups: Union[str, Sequence[str]],
                cropno: Union[str, int, Sequence[str], Sequence[int]],
                metric: str,
                raw_ds: Union[None, str, Sequence[str]] = None,
                tol_distance: int = 40,
                clip_distance: int = 200,
                threshold: int = 127,
                test: bool = False) -> Optional[Dict[str, Any]]:
    """
    Function to find the best result as measured by a given metric.

    Args:
        db: Database with evaluation results and crop information
        label: label for which to determine best configuration
        setups: training setup for which or training setups across which to determine best configuration
                (iteration/iteration+setup)
        cropno: crops for which to look at evaluation results to determine best configuration - typically one for
                validation mode and list of validation crops for testing, excluding the test crop.
        metric: metric to use for finding best configuration
        raw_ds: raw datasets from which predictions could be pulled for the evaluations considered here
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: threshold applied on top of distance predictions to generate binary segmentation
        test: whether to run in test mode

    Returns:
        None if no results were found (when using manual as metric). Otherwise dictionary specifying best found result.
        If no best results are found corresponding values in returned dictionary are set to None.
    """
    if isinstance(raw_ds, str):
        raw_ds = [raw_ds]
    if isinstance(setups, str):
        setups = [setups]
    if isinstance(cropno, str) or isinstance(cropno,int):
        cropno = [cropno]
    if metric == "manual":
        assert len(cropno) == 1, "Manual validation only applicable to a single crop"
        return _best_manual(db, label, setups, cropno[0], raw_ds=raw_ds)
    else:
        return _best_automatic(db, label, setups, cropno, metric, raw_ds=raw_ds, tol_distance=tol_distance,
                               clip_distance=clip_distance, threshold=threshold, test=test)


def get_diff(db: cosem_db.MongoCosemDB,
             label: str,
             setups: Union[str, Sequence[str]],
             cropno: str,
             metric_best: str,
             metric_compare: str,
             raw_ds: Optional[str] = None,
             tol_distance: int = 40,
             clip_distance: int = 200,
             threshold: int = 127,
             test: bool = False) -> Dict[str, Any]:
    """
    Compare two metrics by measuring performance using `metric_compare` but picking the best configuration using
    metric `metric_best`.

    Args:
        db: Database with evaluation results and crop information
        label: label for which to complete this comparison.
        setups: training setup for which or training setups across which to determine best configuration.
        cropno: crops to analyze to determine best configuration and measure performance
        metric_best: Metric to use for finding best configuration (iteration/iteration+setup)
        metric_compare: Metric to use for reporting performance using the best configuration determined by
                        `metric_best`.
        raw_ds: raw datasets from which predictions could be pulled for the evaluations considered here
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: threshold applied on top of distance predictions to generate binary segmentation
        test: whether to run in test mode

    Returns:
        Dictionary with evaluation result measured by `metric_compare` but optimized using `metric_best`.
    """
    best_config = best_result(db, label, setups, cropno, metric_best, raw_ds=raw_ds, tol_distance=tol_distance,
                             clip_distance=clip_distance, threshold=threshold, test=test)
    query_metric2 = best_config.copy()
    query_metric2["metric"] = metric_compare
    query_metric2["metric_params"] = filter_params({"clip_distance": clip_distance, "tol_distance": tol_distance},
                                                   metric_compare)

    if best_config["metric"] != "manual":
        try:
            query_metric2.pop("value")
            query_metric2.pop("_id")
        except KeyError:
            query_metric2["value"] = None
            return query_metric2
    compare_setup = db.find(query_metric2)[0]
    return compare_setup


def _get_csv_files(csv_folder_manual: str, domain: str, cropno: Sequence[Union[int, str]],
                   db: cosem_db.MongoCosemDB) -> List[str]:
    if cropno is None:
        csv_result_files = os.listdir(csv_folder_manual)
        csv_result_files = [fn for fn in csv_result_files if fn.endswith("_{0:}.csv".format(domain))]
    else:
        csv_result_files = []
        for cno in cropno:
            crop = db.get_crop_by_number(cno)
            csv_result_files.append(os.path.join(csv_folder_manual, crop["dataset_id"] + "_{0:}.csv".format(domain)))
    return csv_result_files


def _get_setup_queries(cropno: Sequence[Union[int, str]],
                       db: cosem_db.MongoCosemDB) -> List[Dict[str, Union[str, Sequence[str]]]]:
    csv_folder_manual = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], db.training_version,
                                     "manual")
    csv_result_files = _get_csv_files(csv_folder_manual, "setup", cropno, db)
    setup_queries = []
    for csv_f in csv_result_files:
        f = open(os.path.join(csv_folder_manual, csv_f), "r")
        fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
        cell_id = re.split("_(setup|iteration).csv", csv_f)[0]
        crop = db.get_validation_crop_by_cell_id(cell_id)

        reader = csv.DictReader(f, fieldnames)
        for row in reader:
            if any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in
                   hierarchy[row["labelname"]].labelid):
                # find the csv files with the list of setups compared for each label (4nm or 8nm)
                if row["raw_dataset"] == "volumes/raw/s0":
                    ff = open(os.path.join(csv_folder_manual, "compared_4nm_setups.csv"), "r")
                elif row["raw_dataset"] == "volumes/subsampled/raw/0" or row["raw_dataset"] == "volumes/raw/s1":
                    ff = open(os.path.join(csv_folder_manual, "compared_8nm_setups.csv"), "r")
                else:
                    raise ValueError("The raw_dataset {0:} ".format(row["raw_dataset"]))
                # get that list of compared setups from the csv file
                compare_reader = csv.reader(ff)
                for compare_row in compare_reader:
                    if compare_row[0] == row["labelname"]:
                        setups = compare_row[1:]
                        break
                # collect result
                query = {"label": row["labelname"],
                         "raw_dataset": row["raw_dataset"],
                         "setups": setups,
                         "crop": crop["number"]}
                setup_queries.append(query)
    return setup_queries


def _get_iteration_queries(cropno: Sequence[Union[int, str]], db: cosem_db.MongoCosemDB) -> List[Dict[str, str]]:
    csv_folder_manual = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], db.training_version,
                                     "manual")
    csv_result_files = _get_csv_files(csv_folder_manual, "iteration", cropno, db)
    iteration_queries = []
    for csv_f in csv_result_files:
        f = open(os.path.join(csv_folder_manual, csv_f), "r")
        fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
        cell_id = re.split("_(setup|iteration).csv", csv_f)[0]
        crop = db.get_validation_crop_by_cell_id(cell_id)

        reader = csv.DictReader(f, fieldnames)
        for row in reader:
            if any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in
                   hierarchy[row["labelname"]].labelid):
                query = {"label": row["labelname"],
                         "raw_dataset": row["raw_dataset"],
                         "setups": [row["setup"]],
                         "crop": crop["number"]}
                iteration_queries.append(query)
    return iteration_queries


def get_manual_comparisons(db: cosem_db.MongoCosemDB,
                           cropno: Union[None, str, int, Sequence[Union[str, int]]] = None,
                           mode: str = "across-setups") -> \
        List[Union[Dict[str, str], Dict[str, Union[str, Sequence[str]]]]]:
    """
    Read best configurations optimized manually from corresponding csv files and translate into dictionary that can be
    used for queries to the database with automatic evaluations.

    Args:
        db: Database with crop information
        cropno: Specific crop numbers or list of crop numbers that should be included in queries.
        mode: "per-setup" for queries specifying the optimized manual iteration for each setup, "across-setups"
        (default) for queries specifying the optimized manual iteration and setup for each label and "all" for both.

    Returns: List of corresponding queries.
    """

    if isinstance(cropno, int) or isinstance(cropno, str):
        cropno = [cropno]
    if mode == "across-setups":
        all_queries = _get_setup_queries(cropno, db)
    elif mode == "per-setup":
        all_queries = _get_iteration_queries(cropno, db)
    elif mode == "all":
        all_queries = _get_iteration_queries(cropno, db) + _get_setup_queries(cropno, db)
    else:
        raise ValueError("Unknown mode {mode:}".format(mode=mode))
    return all_queries


def get_refined_comparisons(db: cosem_db.MongoCosemDB,
                            cropno: Union[None, str, int, Sequence[Union[str, int]]] = None) -> List[
    Dict[str, Any]]:
    """
    Get list of queries for predictions that have been refined (as read from csv file)

    Args:
        db: Database with crop information.
        cropno: Specific crop number or list of crop numbers that should be included in queries.

    Returns:
        List of queries for which refined predictions exist.
    """
    csv_folder_refined = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], db.training_version,
                                     "refined")
    # get list of csv files relevant for crops
    if cropno is None:
        csv_result_files = os.listdir(csv_folder_refined)
    else:
        if isinstance(cropno, str) or isinstance(cropno, int):
            cropno = [cropno]
        csv_result_files = []
        for cno in cropno:
            crop = db.get_crop_by_number(cno)
            csv_result_files.append(os.path.join(csv_folder_refined, crop["dataset_id"] + "_setup.csv"))

    # collect entries from those csv files
    queries = []
    for csv_f in csv_result_files:
        f = open(os.path.join(csv_folder_refined, csv_f), "r")
        fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
        cell_id = re.split("_setup.csv", csv_f)[0]
        crop = db.get_validation_crop_by_cell_id(cell_id)

        reader = csv.DictReader(f, fieldnames)
        for row in reader:
            # only consider results that we can evaluate automatically (actually contained in the crop)
            if any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in
                   hierarchy[row["labelname"]].labelid):
                query = {"label": row["labelname"],
                         "raw_dataset": row["raw_dataset"],
                         "setup": row["setup"],
                         "crop": crop["number"],
                         "iteration": int(row["iteration"])}
                queries.append(query)
    return queries


def compare_evaluation_methods(db: cosem_db.MongoCosemDB,
                               metric_compare: str,  # may not be manual
                               metric_bestby: str,  # may be manual
                               queries: List[Union[Dict[str, str], Dict[str, Union[str, Sequence[str]]]]],
                               tol_distance: int = 40,
                               clip_distance: int = 200,
                               threshold: int = 127,
                               test: bool = False) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Compare different metrics for evaluation by picking one metric (`metric_compare`) to report results and optimizing
    the configuration (iteration/iteration+setup) with that metric on the one hand and the metric `metric_bestby` on
    the other hand.

    Args:
        db: Database with crop information and evaluation results.
        metric_compare: Metric to use for reporting performance - using the best configuration determined by
                        `metric_bestby` compared to this metric.
        metric_bestby: Metric to use for finding best configuration (iteration/iteration+setup)
        queries: List of queries for which to compare metrics.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: threshold applied on top of distance predictions to generate binary segmentation
        test: whether to run in test mode

    Returns:
        List of Tuples with evaluation result (reported via `metric_compare`). The first entry will be optimized
        directly for `metric_compare`, the second entry will be optimized for `metric_bestby`.
    """
    comparisons = []
    for qu in queries:
        for setup in qu["setups"]:
            test_query = {"setup": setup,
                          "crop": qu["crop"],
                          "label": qu["label"],
                          "raw_dataset": qu["raw_dataset"],
                          "metric": {"$in": [metric_compare, metric_bestby]}}
            if len(db.find(test_query)) == 0:
                raise RuntimeError("No results found in database for {0:}".format(test_query))
        best_setup = best_result(db, qu["label"], qu["setups"], qu["crop"], metric_compare, raw_ds=qu["raw_dataset"],
                                 tol_distance=tol_distance, clip_distance=clip_distance, threshold=threshold,
                                 test=test)
        compare_setup = get_diff(db, qu["label"], qu["setups"], qu["crop"], metric_bestby, metric_compare,
                                 raw_ds=qu["raw_dataset"], tol_distance=tol_distance, clip_distance=clip_distance,
                                 threshold=threshold, test=test)
        comparisons.append((best_setup, compare_setup))
    return comparisons


def compare_refined(db: cosem_db.MongoCosemDB,
                    metric: str,
                    queries: Sequence[Dict[str, Any]],
                    tol_distance: int = 40,
                    clip_distance: int = 200,
                    threshold: int = 127) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    For given queries read corresponding refined and unrefined results from the database for the given metric.

    Args:
        db: Database with crop information and evaluation results.
        metric: Metrics to use for comparison.
        queries: List of queries for which to compare results for refinements.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: threshold applied on top of distance predictions to generate binary segmentation

    Returns:
        List of tuples with evaluation results. The first entry will be the result before refinements, the second after
        refinements.
    """
    comparisons = []
    for qu in queries:
        qu["metric"] = metric
        qu["metric_params"] = filter_params({"clip_distance": clip_distance, "tol_distance": tol_distance},
                                            metric)
        qu["refined"] = True
        refined = db.find(qu)
        assert len(refined) == 1, f"len(refined)={len(refined)}, qu: {qu}"
        refined = refined[0]
        qu["refined"] = False
        qu["threshold"] = threshold
        not_refined = db.find(qu)
        if len(not_refined) != 1:
            print([x for x in not_refined])
        assert len(not_refined) == 1
        not_refined = not_refined[0]
        comparisons.append((not_refined, refined))
    return comparisons


def compare_setups(db: cosem_db.MongoCosemDB,
                   setups_compare: Sequence[Sequence[str]],
                   labels: Sequence[str],
                   metric: str,
                   raw_ds: Optional[Sequence[str]] = None,
                   crops: Optional[Sequence[Union[str, int]]] = None,
                   tol_distance: int = 40,
                   clip_distance: int = 200,
                   threshold: int = 127,
                   mode: str = "across-setups",
                   test: bool = False) -> List[List[Optional[Dict[str, Any]]]]:
    """
    Flexibly query comparisons from the evaluation database. `setups_compare` and optionally `raw_ds` define sets of
    settings that should be compared.

    Args:
        db: Database with crop information and evaluation results.
        setups_compare: List of list of setups to compare.
        labels: List of labels. In a `mode` = "per-setup" evaluation, these are paired with the entries of the entries
                in each list of `setups_compare`.
        metric: Metric to evaluate for comparing the setups.
        raw_ds: List of raw datasets to consider for querying pulled predictions, can be None if it doesn't matter.
        crops: List of crop numbers to evaluate for. If None it'll be all validation crops
        tol_distance: Tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: Clip distance when using a metric with clip distance, otherwise not used.
        threshold: Threshold applied on top of distance predictions to generate binary segmentation.
        mode: "across-setups" or "per"setup" depending on whether the configuration that should be optimized is both
              the setup and the iteration ("across-setups") or just the iteration for a given setup ("per-setup"_
        test: whether to run in test mode

    Returns:
        List of comparisons. Each entry corresponds to a cropno and label and each entry is itself a list with entries
        corresponding to the each list in `setups_compare` and optionally `raw_ds`.
    """
    comparisons = []
    if crops is None:
        crops = [c["number"] for c in db.get_all_validation_crops()]

    if mode == "across-setups":  # for one label find best result across setups
        for cropno in crops:
            for lbl in labels:
                comp = []
                for k, setups in enumerate(setups_compare):
                    if raw_ds is None:
                        rd = None
                    else:
                        rd = raw_ds[k]
                    comp.append(
                        best_result(db, lbl, setups, cropno, metric, raw_ds=rd, tol_distance=tol_distance,
                                    clip_distance=clip_distance, threshold=threshold, test=test)
                    )
                comparisons.append(comp)
    elif mode == "per-setup":  # find best result for each combination of setup and label
        for cropno in crops:
            comps = [[] for _ in labels]
            for k, setups in enumerate(setups_compare):
                if raw_ds is None:
                    rd = None
                else:
                    rd = raw_ds[k]
                for kk, (lbl, setup) in enumerate(zip(labels, setups)):
                    comps[kk].append(
                        best_result(db, lbl, setup, cropno, metric, raw_ds=rd, tol_distance=tol_distance,
                                    clip_distance=clip_distance, threshold=threshold, test=test)
                    )
            comparisons.extend(comps)
    return comparisons
