import csv
import os
from CNNectome.utils import cosem_db
from CNNectome.validation.organelles.segmentation_metrics import *
from CNNectome.validation.organelles.run_evaluation import construct_pred_path, pred_path_without_iteration
db_host = "cosem.int.janelia.org:27017"
gt_version = "v0003"
training_version = "v0003.2"
csv_folder = "/groups/cosem/cosem/computational_evaluation/{0:}/manual/".format(training_version)


def get_cell_identifier(crop):
    basename, n5_filename = os.path.split(crop["parent"])
    _, cell_identifier = os.path.split(basename)
    return cell_identifier


def get_best_manual(dataset, labelname, setup=None, s1=False):
    if setup is None:
        csv_file = os.path.join(csv_folder, dataset+"_setup.csv")
        f = open(csv_file, "r")
        fieldnames = ["labelname", "setup", "iteration", "s1"]

    else:
        csv_file = os.path.join(csv_folder, dataset+"_iteration.csv")
        f = open(csv_file, "r")
        fieldnames = ["setup", "labelname", "iteration", "s1"]
    reader = csv.DictReader(f, fieldnames)
    for row in reader:
        if setup is None or row["setup"] == setup:
            if row["labelname"] == labelname:
                if bool(int(row["s1"])) == s1:
                    result = row
                    break

    iteration = int(result["iteration"])
    setup = row["setup"]
    f.close()
    return labelname, setup, iteration, s1


def query_score(db_username, db_password, cropno, labelname, threshold=127, setup=None, s1=False, clip_distance=200,
                tol_distance=40):
    db = cosem_db.MongoCosemDB(db_username, db_password, host=db_host, gt_version=gt_version, 
                               training_version=training_version)
    c = db.get_crop_by_number(cropno)
    cell_identifier = get_cell_identifier(c)
    labelname, setup, iteration, s1 = get_best_manual(cell_identifier, labelname, setup=setup, s1=s1)
    path = construct_pred_path(setup, iteration, c, s1)
    threshold = threshold
    metric_params = dict()
    metric_params["clip_distance"] = clip_distance
    metric_params["tol_distance"] = tol_distance
    scores = dict()
    for metric in EvaluationMetrics:
        specific_params = filter_params(metric_params, metric)
        query = {"path": path, "dataset": labelname, "setup": setup, "iteration": iteration, "crop": str(cropno),
                 "threshold": threshold, "metric": metric, "metric_params": specific_params}
        doc = db.read_evaluation_result(query)
        scores[metric.name] = doc["value"]

    return scores


def get_best_automatic(db, cropno, labelname, metric, metric_params, setup=None, threshold=127, s1=False):
    metric_params = filter_params(metric_params, metric)
    query = {"label": labelname, "threshold": threshold, "crop": str(cropno), "metric": metric, "metric_params":
             metric_params}

    if setup is not None:
        query["setup"] = setup
        query["path"] = {"$regex": pred_path_without_iteration(setup, db.get_crop_by_number(cropno), s1)}
    result_set = db.find(query)
    values = [r["value"] for r in result_set]
    best_arg = best(metric)(values)
    return result_set[best_arg]


def get_differences(db_username, db_password, cropno, metrics, domain="setup", threshold=127, clip_distance=200,
                    tol_distance=40):
    if not (isinstance(metrics, tuple) or isinstance(metrics, list)):
        metrics = [metrics, ]
    db = cosem_db.MongoCosemDB(db_username, db_password, host=db_host, gt_version=gt_version,
                               training_version=training_version)
    print(str(cropno))
    c = db.get_crop_by_number(str(cropno))
    print(c)
    metric_params = dict()
    metric_params["clip_distance"] = clip_distance
    metric_params["tol_distance"] = tol_distance

    cell_identifier = get_cell_identifier(c)
    if domain == "setup":
        csv_file = os.path.join(csv_folder, cell_identifier+"_setup.csv")
        f = open(csv_file, "r")
        fieldnames = ["labelname", "setup", "iteration", "s1"]
    elif domain == "iteration":
        csv_file = os.path.join(csv_folder, cell_identifier+"_iteration.csv")
        f = open(csv_file, "r")
        fieldnames = ["setup", "labelname", "iteration", "s1"]
    else:
        raise ValueError("unknown domain")

    reader = csv.DictReader(f, fieldnames)
    all_manual = []
    for row in reader:
        manual_result = {}
        manual_result["setup"] = row["setup"]
        manual_result["labelname"] = row["labelname"]
        manual_result["iteration"] = int(row["iteration"])
        manual_result["s1"] = bool(int(row["s1"]))
        all_manual.append(manual_result)
    f.close()
    for manual_result in all_manual:
        if domain == "setup":
            query_setup = None
        else:
            query_setup = manual_result["setup"]

        for best_by_metric in metrics + ["manual"]:

            for eval_metric in metrics:

                entry = "{eval_metric:}_by_{best_by_metric:}".format(eval_metric=eval_metric, best_by_metric=best_by_metric)
                if best_by_metric == "manual":
                    manual_result[entry] = query_score(db_username, db_password, cropno, manual_result["labelname"],
                                                     setup=query_setup, s1=manual_result["s1"],
                                                     threshold=threshold, clip_distance=clip_distance,
                                                     tol_distance=tol_distance)[eval_metric]
                else:
                    specific_params = filter_params(metric_params, best_by_metric)
                    best_result = get_best_automatic(db, cropno, manual_result["labelname"], best_by_metric, specific_params,
                                                     query_setup, threshold=threshold, s1=manual_result["s1"])

                    specific_params = filter_params(metric_params, eval_metric)
                    query = {"path": best_result["path"],
                             "dataset": best_result["dataset"],
                             "setup": best_result["setup"],
                             "iteration": best_result["iteration"],
                             "label": best_result["label"],
                             "crop": best_result["crop"],
                             "threshold": best_result["threshold"],
                             "metric": eval_metric,
                             "metric_params": specific_params}
                    doc = db.read_evaluation_result(query)
                    manual_result[entry] = doc["value"]

    return all_manual


if __name__ == "__main__":
    print(get_differences("root", "root", 111, "f1_score", domain="iteration"))
