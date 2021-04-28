from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.utils import config_loader
from CNNectome.utils.hierarchy import *
import argparse
import os


def run_new_crop(cropno, gt_version="v0003", training_version="v0003.2", tol_distance=40, clip_distance=200):
    db = cosem_db.MongoCosemDB(write_access=True, gt_version=gt_version, training_version=training_version)
    eval_results_csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"],
                                           training_version,
                                           "evaluation_results")
    csvhandler = cosem_db.CosemCSV(eval_results_csv_folder)
    col = db.access("evaluation", db.training_version)
    for k, entry in enumerate(col.find({"crop": {"$ne": str(cropno)}, "label": {"$ne": "ribosomes"}}, batch_size=100)):
        print(".", end="", flush=True)
        setup = entry["setup"]
        iteration = entry["iteration"]
        s1 = "s1" in entry["raw_dataset"]
        crop = db.get_crop_by_number(cropno)
        pred_path = construct_pred_path(setup, iteration, crop, s1, training_version=training_version)
        test_query = {"path": pred_path, "dataset": entry["dataset"], "setup": setup, "iteration": iteration,
                      "label": entry["label"], "crop": str(cropno), "threshold": entry["threshold"],
                      "metric": entry["metric"], "metric_params": entry["metric_params"]}
        doc = db.read_evaluation_result(test_query)

        if doc is not None:
            continue

        # run evaluations for all metrics together (saves computation time)
        if (entry["metric_params"] == {}) or (entry["metric_params"] == {"tol_distance": tol_distance}) or (
                entry["metric_params"] == {"clip_distance": clip_distance}):
            metric_query = {"path": entry["path"], "dataset": entry["dataset"], "setup": setup,
                            "iteration": iteration, "label": entry["label"], "crop": entry["crop"],
                            "threshold": entry["threshold"]}
            results = db.find(metric_query)
            metrics = []
            metric_params = {"tol_distance": tol_distance, "clip_distance": clip_distance}
            for r in results:
                metrics.append(r["metric"])

            print("\n" + str(k))
            run_validation(pred_path, entry["dataset"], setup, iteration, hierarchy[entry["label"]],
                           db.get_crop_by_number(str(cropno)), entry["threshold"], metrics, metric_params, db,
                           csvhandler, True, False, gt_version=gt_version)


def main():
    parser = argparse.ArgumentParser("Run evaluations for a newly added validation crop")
    parser.add_argument("--crop", type=int, default=111,
                        help="number of crop with annotated groundtruth, e.g. 110")
    parser.add_argument("--tol_distance", type=int, default=40,
                        help="Parameter used for counting false negatives/positives with a tolerance. Only false "
                             "predictions that are farther than this value from the closest pixel where they would be "
                             "correct are counted.")
    parser.add_argument("--clip_distance", type=int, default=200,
                        help="Parameter used for clipped false distances. False distances larger than the value of "
                             "this parameter are reduced to this value.")
    parser.add_argument("--training_version", type=str, default="v0003.2", help="Version of trainings for which to "
                                                                                "run evaluation")
    parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth to use for "
                                                                        "evaluation")
    args = parser.parse_args()
    run_new_crop(args.crop, gt_version=args.gt_version, training_version=args.training_version, tol_distance=
    args.tol_distance, clip_distance=args.clip_distance)


if __name__ == "__main__":
    main()
