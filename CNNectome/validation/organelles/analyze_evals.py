from CNNectome.utils.crop_utils import get_label_ids_by_category, get_cell_identifier
from CNNectome.validation.organelles.segmentation_metrics import *
from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.utils import cosem_db
import csv
import os
import re
from CNNectome.utils.hierarchy import hierarchy

db_host = "cosem.int.janelia.org:27017"
gt_version = "v0003"
training_version = "v0003.2"
csv_folder = "/groups/cosem/cosem/computational_evaluation/{0:}/manual/".format(training_version)


def check_margins(db, best_res):
    query = best_res.copy()
    query.pop("value")
    query.pop("path")
    query.pop("_id")
    query["iteration"] = {"$gt": best_res["iteration"]}
    higher_iteration = db.find(query)
    query["iteration"] = {"$lt": best_res["iteration"]}
    lower_iteration = db.find(query)
    if len(higher_iteration) <= 1 or len(lower_iteration) <=1:
        return False
    else:
        return True


def best_result(db, label, setups, cropno, metric, raw_ds=None, tol_distance=40, clip_distance=200, threshold=127):
    def best_automatic(db, label, setups, cropno, metric, raw_ds=None, tol_distance=40, clip_distance=200,
                       threshold=127):
        metric_params = dict()
        metric_params["clip_distance"] = clip_distance
        metric_params["tol_distance"] = tol_distance
        filtered_params = filter_params(metric_params, metric)
        query = {"label": label, "crop": str(cropno), "metric": metric, "setup": {"$in": setups}, "metric_params":
                 filtered_parms, "threshold": threshold}
        if raw_ds is not None:
            query["raw_dataset"] = {"$in": raw_ds}
        results = [res for res in db.find(query)]
        if len(results) == 0:
            raise ValueError("No results in database matching query {0:}".format(query))
        scores = [res["value"] for res in results]
        try:
            best_score_arg = best(metric)(scores)
            best_res = results[best_score_arg]
            if not check_margins(db, best_res):
                print("{0:} is at the margin of what has been evaluated".format(best_res))
        except ValueError:
            best_res = query.copy()
            best_res.update({"setup": None, "raw_dataset": raw_ds, "value": None, "iteration": None})
        return best_res

    def best_manual(db, label, setups, cropno, raw_ds=None):
        c = db.get_crop_by_number(str(cropno))
        cell_identifier = get_cell_identifier(c)
        csv_file_iterations = open(os.path.join(csv_folder, cell_identifier + "_iteration.csv"), "r")
        fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
        reader = csv.DictReader(csv_file_iterations, fieldnames)

        # look for all possible matches
        best_manuals = []
        for row in reader:
            if row["labelname"] == label and row["setup"] in setups:
                if raw_ds is None or row["raw_dataset"] in raw_ds:
                    manual_result = {"setup": row["setup"], "label": row["labelname"],
                                     "iteration": int(row["iteration"]), "raw_dataset": row["raw_dataset"],
                                     "crop": str(cropno), "metric": "manual"}
                    best_manuals.append(manual_result)

        if len(best_manuals) == 0:  # no manual evaluations with the given constraints
            return None
        elif len(best_manuals) == 1:  # if there's only one match it has to be the best one
            return best_manuals[0]
        else:  # if there's several matches check the setup results for overall best
            csv_file_setups = open(os.path.join(csv_folder, cell_identifier + "_setup.csv"), "r")
            reader = csv.DictReader(csv_file_setups, fieldnames)
            for row in reader:
                if row["labelname"] == label and row["setup"] in setups:
                    if raw_ds is None or row["raw_dataset"] == raw_ds:
                        manual_result_best = {"setup": row["setup"], "label": row["labelname"],
                                              "iteration": int(row["iteration"]), "raw_dataset": row["raw_dataset"],
                                              "crop": str(cropno), "metric": "manual"}
                        return manual_result_best
            return None
    if isinstance(raw_ds, str):
        raw_ds = [raw_ds]
    if isinstance(setups, str):
        setups = [setups]
    if metric == "manual":
        return best_manual(db, label, setups, cropno, raw_ds=raw_ds)
    else:
        return best_automatic(db, label, setups, cropno, metric, raw_ds=raw_ds, tol_distance=tol_distance,
                              clip_distance=clip_distance, threshold=threshold)


def get_diff(db, label, setups, cropno, metric_best, metric_compare, raw_ds=None, tol_distance=40, clip_distance=200,
             threshold=127):
    best_setup = best_result(db, label, setups, cropno, metric_best, raw_ds=raw_ds, tol_distance=tol_distance,
                             clip_distance=clip_distance, threshold=threshold)
    query_metric2 = best_setup.copy()
    query_metric2["metric"] = metric_compare
    query_metric2["metric_params"] = filter_params({"clip_distance": clip_distance, "tol_distance": tol_distance},
                                                   metric_compare)
    if best_setup["metric"] != "manual":
        query_metric2.pop("value")
        query_metric2.pop("_id")
    compare_setup = db.find(query_metric2)[0]
    return compare_setup


def get_manual_comparisons(db, cropno=None, domain=None):
    def get_csv_files(domain):
        if cropno is None:
            csv_result_files = os.listdir(csv_folder)
            csv_result_files = [fn for fn in csv_result_files if fn.endswith("_{0:}.csv".format(domain))]
        else:
            cell_identifier = get_cell_identifier(db.get_crop_by_number(cropno))
            csv_result_files = [os.path.join(csv_folder, cell_identifier + "_{0:}.csv".format(domain))]
        return csv_result_files

    def get_iteration_queries():
        csv_result_files = get_csv_files("iteration")
        iteration_queries = []
        for csv_f in csv_result_files:
            f = open(os.path.join(csv_folder, csv_f), "r")
            fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
            cell_id = re.split("_(setup|iteration).csv", csv_f)[0]
            crop = db.get_validation_crop_by_cell_id(cell_id)

            reader = csv.DictReader(f, fieldnames)
            for row in reader:
                if any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in
                       hierarchy[row["labelname"]].labelid):
                    query = {"label": row["labelname"], "raw_dataset": row["raw_dataset"], "setups": [row["setup"]],
                             "crop": crop["number"]}
                    iteration_queries.append(query)
        return iteration_queries

    def get_setup_queries():
        csv_result_files = get_csv_files("setup")
        setup_queries = []
        for csv_f in csv_result_files:
            f = open(os.path.join(csv_folder, csv_f), "r")
            fieldnames = ["setup", "labelname", "iteration", "raw_dataset"]
            cell_id = re.split("_(setup|iteration).csv", csv_f)[0]
            print(cell_id)
            crop = db.get_validation_crop_by_cell_id(cell_id)

            reader = csv.DictReader(f, fieldnames)
            for row in reader:
                if any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in
                       hierarchy[row["labelname"]].labelid):
                    ff = open(os.path.join(csv_folder, "compared_setups.csv"), "r")
                    compare_reader = csv.reader(ff)
                    for compare_row in compare_reader:
                        if compare_row[0] == row["labelname"]:
                            setups = compare_row[1:]
                            break
                    query = {"label": row["labelname"], "raw_dataset": row["raw_dataset"], "setups": setups,
                             "crop": crop["number"]}
                    setup_queries.append(query)
        return setup_queries

    if domain is None:
        domain = ["setup", "iteration"]
    elif not (isinstance(domain, list) or isinstance(domain, tuple)):
        domain = [domain]
    all_queries = []
    if "setup" in domain:
        all_queries.extend(get_setup_queries())
    if "iteration" in domain:
        all_queries.extend(get_iteration_queries())
    return all_queries




def compare_evaluation_methods(db, metric_compare, metric_bestby, queries, tol_distance=40, clip_distance=200,
                               threshold=127, test=False):
    comparisons = []
    for qu in queries:
        for setup in qu["setups"]:
            test_query = {"setup": setup, "crop": qu["crop"], "label": qu["label"], "raw_dataset": qu["raw_dataset"],
                          "metric": {"$in": [metric_compare, metric_bestby]}}
            if len(db.find(test_query)) == 0:
                raise RuntimeError("No results found in database for {0:}".format(test_query))
        best_setup = best_result(db, qu["label"], qu["setups"], qu["crop"], metric_compare, raw_ds=qu["raw_dataset"],
                                 tol_distance=tol_distance, clip_distance=clip_distance, threshold=threshold)
        compare_setup = get_diff(db, qu["label"], qu["setups"], qu["crop"], metric_bestby, metric_compare,
                                 raw_ds=qu["raw_dataset"], tol_distance=tol_distance, clip_distance=clip_distance,
                                 threshold=threshold)
        comparisons.append((best_setup, compare_setup))
    return comparisons


def compare_setups(db, setups_compare, labels, metric, raw_ds=None, crops=None, tol_distance=40,
                   clip_distance=200, threshold=127, mode="across_setups"):
    comparisons = []
    if crops is None:
        crops = [c["number"] for c in db.get_all_validation_crops()]

    if mode == "across_setups":
        for cropno in crops:
            for lbl in labels:
                comp = []
                for k, setups in enumerate(setups_compare):
                    if raw_ds is None:
                        rd = None
                    else:
                        rd = raw_ds[k]
                    comp.append(best_result(db, lbl, setups, cropno, metric, raw_ds=rd, tol_distance=tol_distance,
                                clip_distance=clip_distance, threshold=threshold))
                comparisons.append(comp)
    elif mode == "per_setup":
        for cropno in crops:
            comps = []
            for lbl in labels:
                comps.append([])
            for k, setups in enumerate(setups_compare):
                if raw_ds is None:
                    rd = None
                else:
                    rd = raw_ds[k]
                for m, (lbl, setup) in enumerate(zip(labels, setups)):
                    comps[m].append(best_result(db, lbl, setup, cropno, metric, raw_ds=rd, tol_distance=tol_distance,
                                clip_distance=clip_distance, threshold=threshold))
            comparisons.extend(comps)
    return comparisons


if __name__ == "__main__":
    import CNNectome.utils.cosem_db
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str)
    parser.add_argument("crop", type=int)
    parser.add_argument("metric", type=str, choices=list(em.value for em in segmentation_metrics.EvaluationMetrics))
    parser.add_argument("--setup", type=str, default=None)
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    args = parser.parse_args()

    db = CNNectome.utils.cosem_db.MongoCosemDB(args.db_username, args.db_password)

    print(best_result(db, args.label, [args.setup], args.crop, args.metric, "volumes/raw"))
