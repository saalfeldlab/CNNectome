from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.utils.hierarchy import *
import argparse
import os


def run_new_crop(cropno, db_username, db_password, db_host, gt_version, training_version, tol_distance=40,
                 clip_distance=200):
    db = cosem_db.MongoCosemDB(db_username, db_password, host=db_host, gt_version=gt_version,
                               training_version=training_version)
    csvhandler = cosem_db.CosemCSV(eval_results_csv_folder)
    col = db.access("evaluation", training_version)
    for k, entry in enumerate(col.find()):
        if int(entry["crop"]) != cropno and entry["label"] != "ribosomes":
            setup = entry["setup"]
            iteration = entry["iteration"]
            s1 = "s1_it" in entry["path"]
            crop = db.get_crop_by_number(cropno)
            pred_path = construct_pred_path(setup, iteration, crop, s1)
            test_query = {"path": pred_path, "dataset": entry["dataset"], "setup": setup, "iteration": iteration,
                          "label": entry["label"], "crop": str(cropno), "threshold": entry["threshold"],
                          "metric": entry["metric"], "metric_params": entry["metric_params"]}
            doc = db.read_evaluation_result(test_query)
            if doc is not None:
                continue

            if (entry["metric_params"] == {}) or (entry["metric_params"] == {"tol_distance": tol_distance}) or (entry["metric_params"] == {"clip_distance": clip_distance}):
                metric_query = {"path": entry["path"], "dataset": entry["dataset"], "setup": setup,
                                "iteration": iteration, "label": entry["label"], "crop": entry["crop"],
                                "threshold":  entry["threshold"]}
                results = db.find(metric_query)
                metrics = []
                metric_params = {"tol_distance": tol_distance, "clip_distance": clip_distance}
                for r in results:
                    metrics.append(r["metric"])
                try:
                    run_validation(pred_path, entry["dataset"], setup, iteration, hierarchy[entry["label"]],
                                   db.get_crop_by_number(str(cropno)), entry["threshold"], metrics, metric_params, db,
                                   csvhandler, True, False)
                except Exception as e:
                    print(k)
                    print(entry)
                    raise e


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--crop", type=int, default=111,
                        help="number of crop with annotated groundtruth, e.g. 110")
    parser.add_argument("--tol_distance", type=int, default=40,
                        help="Parameter used for counting false negatives/positives with a tolerance. Only false "
                             "predictions that are farther than this value from the closest pixel where they would be "
                             "correct are counted.")
    parser.add_argument("--clip_distance", type=int, default=200,
                        help="Parameter used for clipped false distances. False distances larger than the value of "
                             "this parameter are reduced to this value.")
    args = parser.parse_args()
    run_new_crop(args.crop, args.db_username, args.db_password, db_host, gt_version, training_version,
                 args.tol_distance, args.clip_distance)


if __name__ == "__main__":
    main()
