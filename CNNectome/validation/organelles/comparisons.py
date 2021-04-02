import CNNectome.validation.organelles.analyze_evals as analyze_evals
import CNNectome.validation.organelles.segmentation_metrics as segmentation_metrics
from CNNectome.utils import cosem_db, crop_utils, hierarchy
import csv
import argparse
import tabulate
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def compare_4vs8(db: cosem_db.MongoCosemDB,
                 metric: str,
                 crops: Optional[Sequence[Union[str, int]]] = None,
                 tol_distance: int = 40,
                 clip_distance: int = 200,
                 threshold: int = 127,
                 mode: str = "across_setups",
                 test: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Compare 4nm setups with corresponding 8nm setups.
    Args:
        db: Databse with crop information and evaluation result.
        metric: Metric to use for comparison.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: Threshold to have been applied on top of raw predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup
        test: whether to run in test mode

    Returns:
        List of comparisons. Each comparison is itself a list with the first entry the 4nm result and the second entry
        the corresponding 8nm result.
    """
    if mode == "across_setups":
        setups = [("setup03", "setup25", "setup27.1", "setup31", "setup35", "setup45", "setup47"),
                  ("setup04", "setup26.1", "setup28", "setup32", "setup36", "setup46", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane"]

    elif mode == "per_setup":
        setups = [("setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03",
                   "setup03", "setup03",
                   "setup25", "setup25",
                   "setup27.1", "setup27.1",
                   "setup31", "setup31",
                   "setup35",
                   "setup45", "setup45",
                   "setup47", "setup47"),
                  ("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04",
                   "setup26.1", "setup26.1",
                   "setup28", "setup28",
                   "setup32", "setup32",
                   "setup36",
                   "setup46", "setup46",
                   "setup48", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane",
                  "mito", "mito_membrane",
                  "er", "er_membrane",
                  "microtubules", "microtubules_out",
                  "nucleus",
                  "ecs", "plasma_membrane",
                  "MVB", "MVB_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))

    raw_ds = ["volumes/raw", "volumes/subsampled/raw/0"]
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, raw_ds=raw_ds, crops=crops,
                                               tol_distance=tol_distance, clip_distance=clip_distance,
                                               threshold=threshold, mode=mode, test=test)
    return comparisons


def compare_s1vssub(db: cosem_db.MongoCosemDB,
                    metric: str,
                    crops: Optional[Sequence[Union[str, int]]] = None,
                    tol_distance: int = 40,
                    clip_distance: int = 200,
                    threshold: int = 127,
                    mode: str = "across_setups",
                    test: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Compare 8nm setups run on averaged downsampled data to the same setup run on randomly subsampled data.
    Args:
        db: Databse with crop information and evaluation result.
        metric: Metric to use for comparison.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: Threshold to have been applied on top of raw predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup
        test: whether to run in test mode

    Returns:
        List of comparisons. Each comparison is itself a list with the first entry the result when run on averaged
        downsampled data and the second entry the corresponding result when run on randomly subsampled data.
    """
    raw_ds = ["volumes/raw/s1", "volumes/subsampled/raw/0"]

    if mode == "across_setups":
        setups = [("setup04", "setup26.1", "setup28", "setup32", "setup46", "setup48"),
                  ("setup04", "setup26.1", "setup28", "setup32", "setup46", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane"]
    elif mode == "per_setup":
        setups = [("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04", "setup04",
                   "setup26.1", "setup26.1",
                   "setup28", "setup28",
                   "setup32", "setup32",
                   "setup36",
                   "setup46", "setup46",
                   "setup48", "setup48"),
                  ("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04", "setup04",
                   "setup26.1", "setup26.1",
                   "setup28", "setup28",
                   "setup32", "setup32",
                   "setup36",
                   "setup46", "setup46",
                   "setup48", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane",
                  "mito", "mito_membrane",
                  "er", "er_membrane",
                  "microtubules", "microtubules_out",
                  "nucleus",
                  "ecs", "plasma_membrane",
                  "MVB", "MVB_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, raw_ds=raw_ds, crops=crops,
                                               tol_distance=tol_distance, clip_distance=clip_distance,
                                               threshold=threshold, mode=mode, test=test)
    return comparisons


def compare_allvscommonvssingle_4nm(db: cosem_db.MongoCosemDB,
                                    metric: str,
                                    crops: Optional[Sequence[Union[str, int]]] = None,
                                    tol_distance: int = 40,
                                    clip_distance: int = 200,
                                    threshold: int = 127,
                                    test: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Compare 4nm setups trained on all labels, many labels and few labels.
    Args:
        db: Databse with crop information and evaluation result.
        metric: Metric to use for comparison.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: Threshold to have been applied on top of raw predictions.
        test: whether to run in test mode

    Returns:
        List of comparisons. Each comparison is itself a list with the first entry the result with the setup trained
        on all labels, the second entry the result with the setup trained on many labels, the third entry the result
        with setups each trained on just a few labels.
    """
    setups = [("setup01",),
              ("setup03",),
              ("setup25", "setup27.1", "setup31", "setup35", "setup45", "setup47")]
    labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
              "plasma_membrane", "MVB", "MVB_membrane"]
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, crops=crops, tol_distance=tol_distance,
                                               clip_distance=clip_distance, threshold=threshold, test=test)
    return comparisons


def compare_metrics(db: cosem_db.MongoCosemDB,
                    metric_compare: str,
                    metric_bestby: str,
                    crops: Union[None, str, int, Sequence[Union[str, int]]] = None,
                    tol_distance: int = 40,
                    clip_distance: int = 200,
                    threshold: int = 127,
                    mode: str = "across_setups",
                    test: bool = False) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    A way of comparing the effect of choosing different metrics. Optimize iteration ("per_setup") / setup+iteration
    ("across_setups") using both metrics, then compare those evaluations using one of the metrics (`metric_compare).
    The `metric_bestby` can be manual and comparisons will always be run for the configurations for which a manual
    evaluation has been completed.

    Args:
        db: Database with crop information and evaluation results.
        metric_compare: Metric used to report comparisons.
        metric_bestby: Metric to compare to, used to optimize configuration.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using a metric with clip distance, otherwise not used.
        threshold: Threshold to have been applied on top of raw predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup
        test: whether to run in test mode.

    Returns:
        List of comparisons. Each comparison is a tuple with the first entry the result with the configuration
        optimized for `metric_compare`, the second entry optimized for `metric_bestby`.
    """
    all_queries = analyze_evals.get_manual_comparisons(db, cropno=crops, mode=mode)
    comparisons = analyze_evals.compare_evaluation_methods(db, metric_compare, metric_bestby, all_queries,
                                                           tol_distance=tol_distance, clip_distance=clip_distance,
                                                           threshold=threshold, test=test)
    return comparisons


def compare_rawvsrefined(db: cosem_db.MongoCosemDB, 
                         metric: str, 
                         crops: Union[None, str, int, Sequence[Union[str, int]]] = None, 
                         tol_distance: int = 40, 
                         clip_distance: int = 200, 
                         threshold: int = 127) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Evaluate effect of refinements.
    Args:
        db: Database with crop information and evaluation result.
        metric: Metric to use for comparison.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used
        clip_distance: clip distance when using a metric with clip distance, otherwise not used
        threshold: Threshold to have been applied on top of raw predictions.

    Returns:
        List of comparisons. Each comparison is a tuple with the first entry the result before refinements, the second
        after refinements.
    """
    all_queries = analyze_evals.get_refined_comparisons(db, cropno=crops)
    comparisons = analyze_evals.compare_refined(db, metric, all_queries, tol_distance=tol_distance,
                                                clip_distance=clip_distance, threshold=threshold)
    return comparisons


def best_4nm(db: cosem_db.MongoCosemDB, 
             metric: str, 
             crops: Optional[Sequence[Union[str, int]]] = None, 
             tol_distance: int = 40, 
             clip_distance: int = 200, 
             threshold: int = 200, 
             mode: str = "across_setups",
             raw_ds: Union[None, str, Sequence[str]] = "volumes/raw",
             test: bool = False) -> List[List[Optional[Dict[str, Any]]]]:
    """
    Get the best results for the 4nm setups.

    Args:
        db: Database with crop information and evaluation result.
        metric: Metric to report and use for optimiation of iteration/setup.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using a metric with clip distance, otherwise not used.
        threshold: Threshold to have been applied on top of raw predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup.
        raw_ds: raw dataset to run prediction on.
        test: whether to run in test mode.

    Returns:
        List of best results. Each result is a list with just one dictionary.
    """
    if mode == "across_setups":
        setups = ["setup01", "setup03", "setup25", "setup27.1", "setup31", "setup33", "setup35", "setup45", "setup47",
                  "setup56", "setup59"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "mito_DNA", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "lysosome", "lysosome_membrane", "er", "er_membrane", "ERES", "nucleus", "NE",
                  "NE_membrane", "nuclear_pore", "nuclear_pore_out", "chromatin", "EChrom", "microtubules",
                  "microtubules_out", "ribosomes"]
    elif mode == "per_setup":
        setups = ["setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01",
                  "setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01", "setup01",
                  "setup01", "setup01", "setup01", "setup01", "setup01", "setup01","setup03", "setup03", "setup03",
                  "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03",
                  "setup03", "setup03", "setup25", "setup25", "setup25", "setup27.1", "setup27.1", "setup31",
                  "setup31", "setup33", "setup35", "setup45", "setup45", "setup47", "setup47", "setup47", "setup47",
                  "setup56", "setup59", "setup59"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "mito_DNA", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "lysosome", "lysosome_membrane", "er", "er_membrane", "ERES", "nucleus", "NE",
                  "NE_membrane", "nuclear_pore", "nuclear_pore_out", "chromatin", "EChrom", "microtubules",
                  "microtubules_out", "ribosomes", "ecs", "plasma_membrane", "mito", "mito_membrane", "vesicle",
                  "vesicle_membrane", "MVB", "MVB_membrane", "er", "er_membrane", "ERES", "nucleus", "microtubules",
                  "microtubules_out", "mito", "mito_membrane", "mito_DNA", "er", "er_membrane", "microtubules",
                  "microtubules_out", "ribosomes", "nucleus", "ecs", "plasma_membrane", "MVB", "MVB_membrane",
                  "lysosome", "lysosome_membrane", "ribosomes", "vesicle", "vesicle_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))

    results = []
    if crops is None:
        crops = [c["number"] for c in db.get_all_validation_crops()]
    for cropno in crops:
        if mode == "across_setups":
            for lbl in labels:
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, setups, cropno, metric, raw_ds=raw_ds,
                                                              tol_distance=tol_distance, clip_distance=clip_distance,
                                                              threshold=threshold, test=test)])
        elif mode == "per_setup":
            for setup, lbl in zip(setups, labels):
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, [setup], cropno, metric, raw_ds=raw_ds,
                                                              tol_distance=tol_distance, clip_distance=clip_distance,
                                                              threshold=threshold, test=test)])
    return results


def best_8nm(db: cosem_db.MongoCosemDB, 
             metric: str, 
             crops: Optional[Sequence[Union[str, int]]], 
             tol_distance: int = 40, 
             clip_distance: int = 200, 
             threshold: int = 200, 
             mode: str = "across_setups",
             raw_ds: Union[None, str, Sequence[str]] = "volumes/subsampled/raw/0",
             test: bool = False) -> List[List[Dict[str, Any]]]:
    """
    Get the best results for the 8nm setups.

    Args:
        db: Database with crop information and evaluation result.
        metric: Metric to report and use for optimiation of iteration/setup.
        crops: List of crops to run comparison on. If None will use all validation crops.
        tol_distance: tolerance distance when using a metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using a metric with clip distance, otherwise not used.
        threshold: Threshold to have been applied on top of raw predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup.
        raw_ds: raw dataset to run prediction on.
        test: whether to run in test mode.

    Returns:
        List of best results. Each result is a list with just one dictionary.
    """
    if mode == "across_setups":
        setups = ["setup04", "setup26.1", "setup28", "setup32", "setup36", "setup46", "setup48"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "mito_DNA", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "lysosome", "lysosome_membrane", "er", "er_membrane", "ERES", "nucleus",
                  "microtubules", "microtubules_out"]
    elif mode == "per_setup":
        setups = ["setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                  "setup04", "setup04", "setup04", "setup04", "setup04",
                  "setup26.1", "setup26.1", "setup26.1",
                  "setup28", "setup28",
                  "setup32", "setup32",
                  "setup36",
                  "setup46", "setup46",
                  "setup48", "setup48", "setup48", "setup48"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "er", "er_membrane", "ERES", "nucleus", "microtubules", "microtubules_out",
                  "mito", "mito_membrane", "mito_DNA",
                  "er", "er_membrane",
                  "microtubules", "microtubules_out",
                  "nucleus",
                  "ecs", "plasma_membrane",
                  "MVB", "MVB_membrane", "lysosome", "lysosome_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))
    
    results = []
    if crops is None:
        crops = [c["number"] for c in db.get_all_validation_crops()]
    for cropno in crops:
        if mode == "across_setups":
            for lbl in labels:
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, setups, cropno, metric, raw_ds=raw_ds,
                                                          tol_distance=tol_distance, clip_distance=clip_distance,
                                                          threshold=threshold, test=test)])
        elif mode == "per_setup":
            for setup, lbl in zip(setups, labels):
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, [setup], cropno, metric, raw_ds=raw_ds,
                                                              tol_distance=tol_distance, clip_distance=clip_distance,
                                                              threshold=threshold, test=test)])
    return results


def print_comparison(comparison_task: str,
                     db: cosem_db.MongoCosemDB,
                     metric: Sequence[str],
                     crops: Optional[Sequence[Union[str, int]]] = None,
                     save: Optional[str] = None,
                     tol_distance: int = 40,
                     clip_distance: int = 200,
                     threshold: int = 127,
                     mode: str = "across_setups",
                     raw_ds: str = "volumes/raw",
                     test: bool = False) -> None:
    """
    Print out table with results for chosen comparison. Not all arguments are relevant to all comparison tasks.
    Args:
        comparison_task: Type of comparison to complete ("4vs8"/"s1vssub"/"rawvsrefined"/"allvscommonvssingle"/
                         "metrics"/"best_4nm"/"best_8nm")
        db: Database with crop information and evaluation results.
        metric: Metrics to use for comparison. List of 2 for "metrics", otherwise list of 1.
        crops: List of crops to run comparison on. If None will use all validation crops.
        save: path to csv file for saving output. If None, result won't be saved.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        threshold: Threshold to habe been applied on top of predictions.
        mode: "across_setups" to optimize both setup+iteration or "per_setup" to optimize iteration for a fixed setup.
        raw_ds: Raw dataset to consider predictions for.
        test: whether to run in test mode.

    Returns:
        None.
    """

    if comparison_task == "4vs8":
        comparisons = compare_4vs8(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                                   threshold=threshold, mode=mode, test=test)
        names = ["_4nm", "_8nm"]
    elif comparison_task == "s1vssub":
        comparisons = compare_s1vssub(db, metric[0], crops=crops, tol_distance=tol_distance,
                                      clip_distance=clip_distance, threshold=threshold, mode=mode, test=test)
        names = ["_s1", "_subsampled"]
    elif comparison_task == "rawvsrefined":
        comparisons = compare_rawvsrefined(db, metric[0], crops=crops, tol_distance=tol_distance,
                                           clip_distance=clip_distance, threshold=threshold)
        names = ["_raw", "_refined"]
    elif comparison_task == "allvscommonvssingle":
        comparisons = compare_allvscommonvssingle_4nm(db, metric[0], crops=crops, tol_distance=tol_distance,
                                                      clip_distance=clip_distance, threshold=threshold,
                                                      test=test)
        names = ["_all", "_common", "_single"]
    elif comparison_task == "metrics":
        comparisons = compare_metrics(db, metric[0], metric[1], crops=crops, tol_distance=tol_distance,
                                      clip_distance=clip_distance, threshold=threshold, mode=mode, test=test)
        names = ["_" + metric[0], "_" + metric[1]]
    elif comparison_task == "best_4nm":
        comparisons = best_4nm(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                               threshold=threshold, mode=mode, raw_ds=raw_ds, test=test)
        names = [""]
    elif comparison_task == "best_8nm":
        comparisons = best_8nm(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                               threshold=threshold, mode=mode, raw_ds=raw_ds, test=test)
        names = [""]
    else:
        raise ValueError("Unknown comparison option {0:}".format(comparison_task))

    # define columns for table
    columns = ["label{0:}", "setup{0:}", "iteration{0:}", "crop{0:}", "raw_dataset{0:}", "metric{0:}",
               "metric_params{0:}", "value{0:}"]
    fields = []
    for name in names:
        fields.extend([col.format(name) for col in columns])
    rows = []
    for comparison in comparisons:
        row = []
        for comp in comparison:
            row.extend([comp["label"], comp["setup"], comp["iteration"], comp["crop"], comp["raw_dataset"],
                        comp["metric"], comp["metric_params"], comp["value"]])
        rows.append(row)
    print(tabulate.tabulate(rows, fields))
    if save is not None:
        compare_writer = csv.writer(open(save, "w"))
        compare_writer.writerow(fields)
        for row in rows:
            compare_writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser("Run a pre-defined comparison for evaluation results in the database.")
    parser.add_argument("comparison", type=str, help="Type of comparison to run",
                        choices=["4vs8", "s1vssub", "rawvsrefined", "allvscommonvssingle", "metrics", "best_4nm",
                                 "best_8nm"])
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--metric", nargs="+", type=str,
                        choices=list(em.value for em in segmentation_metrics.EvaluationMetrics) + ["manual"],
                        help="Metric to use for evaluation. For metrics evaluation the first one is used for "
                             "comparison, the second one is the alternative metric by which to pick the best result"
                        )
    parser.add_argument("--crops", type=int, nargs="*", default=None, help="Crops on which .")
    parser.add_argument("--threshold", type=int, default=127, help="threshold applied on distances for evaluation")
    parser.add_argument("--clip_distance", type=int, default=200, help="Parameter used for clipped false distances "
                                                                       "for relevant metrics.")
    parser.add_argument("--tol_distance", type=int, default=40, help="Parameter used for tolerated false distances "
                                                                     "for relevant metrics.")
    parser.add_argument("--mode", type=str, choices=["across_setups", "per_setup", "all"],
                        help="Mode for some of the comparisons on whether to compare across setups ("
                             "`across_setups`) or only between equivalent setups (`per_setup`)",
                        default="across_setups")
    parser.add_argument("--test", action="store_true", help="use cross validation for automatic "
                                                                            "evaluations")
    parser.add_argument("--raw_ds", type=str, help="filter for raw dataset", default="volumes/raw")
    parser.add_argument("--csv", type=str, default=None, help="csv file to save comparisons to")
    args = parser.parse_args()
    if args.mode == "all" and args.comparison != "metrics":
        raise ValueError("Mode all can only be used for metrics comparison.")
    db = cosem_db.MongoCosemDB(args.db_username, args.db_password)
    print_comparison(args.comparison, db, args.metric, crops=args.crops, save=args.csv, tol_distance=args.tol_distance,
                     clip_distance=args.clip_distance, threshold=args.threshold, mode=args.mode, raw_ds=args.raw_ds,
                     test=args.test)


if __name__ == "__main__":
    main()
