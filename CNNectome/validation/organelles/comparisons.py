import CNNectome.validation.organelles.analyze_evals as analyze_evals
import CNNectome.validation.organelles.segmentation_metrics as segmentation_metrics
from CNNectome.utils import cosem_db, crop_utils, hierarchy
import csv
import argparse
import tabulate

def compare_4vs8(db, metric, crops=None, tol_distance=40, clip_distance=200, threshold=127, mode="across_setups"):
    if mode == "across_setups":
        setups = [("setup03", "setup25", "setup27.1", "setup31", "setup35", "setup45", "setup47"),
                  ("setup04", "setup26.1", "setup28", "setup32", "setup36", "setup46", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane"]

    elif mode == "per_setup":
        setups = [("setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03", "setup03",
                   "setup03", "setup03", "setup25", "setup25", "setup27.1", "setup27.1", "setup31", "setup31",
                   "setup35", "setup45", "setup45", "setup47", "setup47"),
                  ("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04", "setup26.1", "setup26.1", "setup28", "setup28", "setup32", "setup32",
                   "setup36", "setup46", "setup46", "setup48", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane", "mito", "mito_membrane", "er", "er_membrane",
                  "microtubules", "microtubules_out", "nucleus", "ecs", "plasma_membrane", "MVB", "MVB_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))

    raw_ds = ["volumes/raw", "volumes/subsampled/raw/0"]
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, raw_ds=raw_ds, crops=crops,
                                               tol_distance=tol_distance, clip_distance=clip_distance,
                                               threshold=threshold, mode=mode)
    return comparisons


def compare_s1vssub(db, metric, crops=None, tol_distance=40, clip_distance=200, threshold=127, mode="across_setups"):
    raw_ds = ["volumes/raw/s1", "volumes/subsampled/raw/0"]

    if mode == "across_setups":
        setups = [("setup04", "setup26", "setup28", "setup32", "setup46", "setup48"),
                  ("setup04", "setup26", "setup28", "setup32", "setup46", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane"]
    elif mode == "per_setup":
        setups = [("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04", "setup04", "setup26", "setup26", "setup28", "setup28", "setup32", "setup32",
                   "setup36", "setup46", "setup46", "setup48", "setup48"),
                  ("setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                   "setup04", "setup04", "setup04", "setup26", "setup26", "setup28", "setup28", "setup32", "setup32",
                   "setup36", "setup46", "setup46", "setup48", "setup48")]
        labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
                  "plasma_membrane", "MVB", "MVB_membrane", "mito", "mito_membrane", "er", "er_membrane",
                  "microtubules", "microtubules_out", "nucleus", "ecs", "plasma_membrane", "MVB", "MVB_membrane"]
    else:
        raise ValueError("unknown mode {0:}".format(mode))
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, raw_ds=raw_ds, crops=crops,
                                               tol_distance=tol_distance, clip_distance=clip_distance,
                                               threshold=threshold, mode=mode)
    return comparisons


def compare_allvscommonvssingle_4nm(db, metric, crops=None, tol_distance=40, clip_distance=200, threshold=127):
    setups = [("setup01",),
              ("setup03",),
              ("setup25", "setup27.1", "setup31", "setup35", "setup45", "setup47")]
    labels = ["mito", "mito_membrane", "er", "er_membrane", "microtubules", "microtubules_out", "nucleus", "ecs",
              "plasma_membrane", "MVB", "MVB_membrane"]
    comparisons = analyze_evals.compare_setups(db, setups, labels, metric, crops=crops, tol_distance=tol_distance,
                                               clip_distance=clip_distance, threshold=threshold)
    return comparisons


def compare_metrics(db, metric_compare, metric_bestby, crops=None, domain=None, tol_distance=40, clip_distance=200,
                    threshold=127):
    all_queries = analyze_evals.get_manual_comparisons(db, cropno=cropno, domain=domain)
    comparisons = analyze_evals.compare_evaluation_methods(db, metric_compare, metric_bestby, all_queries,
                                                           tol_distance=tol_distance, clip_distance=clip_distance,
                                                threshold=threshold)
    return comparisons


def best_4nm(db, metric, crops, tol_distance=40, clip_distance=200, threshold=200, mode="across_setups",
             raw_ds="volumes/raw"):
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
                                                          threshold=threshold)])
        elif mode == "per_setup":
            for setup, lbl in zip(setups, labels):
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, [setup], cropno, metric, raw_ds=raw_ds,
                                                              tol_distance=tol_distance, clip_distance=clip_distance,
                                                              threshold=threshold)])
    return results


def best_8nm(db, metric, crops, tol_distance=40, clip_distance=200, threshold=200, mode="across_setups",
             raw_ds="volumes/subsampled/raw/0"):
    if mode == "across_setups":
        setups = ["setup04", "setup26.1", "setup28", "setup32", "setup36", "setup46", "setup48"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "mito_DNA", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "lysosome", "lysosome_membrane", "er", "er_membrane", "ERES", "nucleus", "NE",
                  "NE_membrane", "nuclear_pore", "nuclear_pore_out", "chromatin", "EChrom", "microtubules",
                  "microtubules_out", 'ribosomes']
    elif mode == "per_setup":
        setups = ["setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04", "setup04",
                  "setup04", "setup04", "setup04", "setup04", "setup04", "setup26.1", "setup26.1", "setup26.1",
                  "setup28", "setup28", "setup32", "setup32", "setup36", "setup46", "setup46", "setup48", "setup48",
                  "setup48", "setup48"]
        labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "vesicle", "vesicle_membrane", "MVB",
                  "MVB_membrane", "er", "er_membrane", "ERES", "nucleus", "microtubules", "microtubules_out",
                  "mito", "mito_membrane", "mito_DNA", "er", "er_membrane", "microtubules", "microtubules_out",
                  "nucleus", "ecs", "plasma_membrane", "MVB", "MVB_membrane", "lysosome", "lysosome_membrane"]
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
                                                          threshold=threshold)])
        elif mode == "per_setup":
            for setup, lbl in zip(setups, labels):
                if crop_utils.check_label_in_crop(hierarchy.hierarchy[lbl], db.get_crop_by_number(cropno)):
                    results.append([analyze_evals.best_result(db, lbl, [setup], cropno, metric, raw_ds=raw_ds,
                                                              tol_distance=tol_distance, clip_distance=clip_distance,
                                                              threshold=threshold)])
    return results


def print_comparison(comparison_task, db, metric, crops=None, save=None, tol_distance=40, clip_distance=200,
                     threshold=127, mode="across_setups", raw_ds="volumes/raw"):
    columns = ["label{0:}", "setup{0:}", "iteration{0:}", "crop{0:}", "raw_dataset{0:}", "metric{0:}",
               "metric_params{0:}", "value{0:}"]
    if comparison_task == "4vs8":
        comparisons = compare_4vs8(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                                   threshold=threshold, mode=mode)
        names = ["_4nm", "_8nm"]
    elif comparison_task == "s1vssub":
        comparisons = compare_s1vssub(db, metric[0], crops=crops, tol_distance=tol_distance,
                                      clip_distance=clip_distance, threshold=threshold, mode=mode)
        names = ["_s1", "_subsampled"]
    elif comparison_task == "allvscommonvssingle":
        comparisons = compare_allvscommonvssingle_4nm(db, metric[0], crops=crops, tol_distance=tol_distance,
                                                      clip_distance=clip_distance, threshold=threshold)
        names = ["_all", "_common", "_single"]
    elif comparison_task == "metrics":
        comparisons = compare_metrics(db, metric[0], metric[1], crops=crops, tol_distance=tol_distance,
                                      clip_distance=clip_distance, threshold=threshold)
        names = ["_" + metric[0], "_" + metric[1]]
    elif comparison_task == "best_4nm":
        comparisons = best_4nm(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                               threshold=threshold, mode=mode, raw_ds=raw_ds)
        names = [""]
    elif comparison_task == "best_8nm":
        comparisons = best_8nm(db, metric[0], crops=crops, tol_distance=tol_distance, clip_distance=clip_distance,
                               threshold=threshold, mode=mode, raw_ds=raw_ds)
        names = [""]
    else:
        raise ValueError("Unknown comparison option {0:}".format(comparison_task))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=str, help="Type of comparison to run",
                        choices=["4vs8", "s1vssub", "allvscommonvssingle", "metrics", "best_4nm", "best_8nm"])
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--metric", nargs="+", type=str,
                        choices=list(em.value for em in segmentation_metrics.EvaluationMetrics) + ["manual"])
    parser.add_argument("--crops", type=int, nargs="*", default=None,
                        help="Metric to use for evaluation. For metrics evaluation the first one is used for "
                             "comparison, the second one is the alternative metric by which to pick the best result")
    parser.add_argument("--threshold", type=int, default=127, help="threshold applied on distances for evaluation")
    parser.add_argument("--clip_distance", type=int, default=200, help="Parameter used for clipped false distances "
                                                                       "for relevant metrics.")
    parser.add_argument("--tol_distance", type=int, default=40, help="Parameter used for tolerated false distances "
                                                                     "for relevant metrics.")
    parser.add_argument("--mode", type=str, choices=["across_setups", "per_setup"],
                        help="Mode for some of the comparisons on whether to compare across setups ("
                             "`across_setups`) or only between equivalent setups (`per_setup`)",
                        default="across_setups")
    parser.add_argument("--raw_ds", type=str, help="filter for raw dataset", default="volumes/raw")
    parser.add_argument("--csv", type=str, default=None, help="csv file to save comparisons to")
    args = parser.parse_args()
    db = cosem_db.MongoCosemDB(args.db_username, args.db_password)
    print_comparison(args.comparison, db, args.metric, crops=args.crops, save=args.csv,
                     tol_distance=args.tol_distance, clip_distance=args.clip_distance, threshold=args.threshold,
                     mode=args.mode, raw_ds=args.raw_ds)


if __name__ == "__main__":
    main()
