from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.utils import config_loader
from CNNectome.utils.hierarchy import *
import argparse
import json
import os

import zarr


def check_mask_attr(n5_path, dataset, valid_masks):
    f = zarr.open(n5_path, "r")
    try:
        mask_ds = f[dataset].attrs["mask_ds"]
    except KeyError as e:
        return e
    return mask_ds in valid_masks

def construct_legacy_pred_path(setup, iteration, crop, s1, training_version="v0003.2"):
    rel_output = os.path.join(*legacy_dataset_names(new=crop["dataset_id"]))
    rel_output = rel_output +"_s1" if s1 else rel_output
    rel_output = rel_output + f"_it{iteration:}.n5"
    
    for tsp in config_loader.get_config()["organelles"]["training_setups_paths"].split(","):
        setup_path = os.path.join(tsp, training_version, setup)
        if os.path.exists(setup_path):
            pred_path = os.path.join(setup_path, rel_output)
            return pred_path
    raise FileNotFoundError("Have not found location for setup {0:}".format(setup))

def find_pred_path(setup, iteration, new_crop, s1, training_version):
    path_new = construct_pred_path(setup, iteration, new_crop, s1, training_version=training_version)
    if os.path.exists(path_new):
        return path_new
    legacy_path = construct_legacy_pred_path(setup, iteration, new_crop, s1, training_version=training_version)
    if os.path.exists(legacy_path):
        return legacy_path
    else:
        cropno = new_crop["number"]
        s1_str = " on s1" if s1 else ""
        raise FileNotFoundError(f"No prediction found for setup {setup:}@{iteration:} for crop {cropno:}{s1_str:}")

def run_new_crop(new_cropno, ref_cropno, gt_version="v0003", training_version="v0003.2", tol_distance=40, 
                 clip_distance=200, setup=None):
    db = cosem_db.MongoCosemDB(write_access=True, gt_version=gt_version, training_version=training_version)
    eval_results_csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"],
                                           training_version, gt_version,
                                           "evaluation_results")
    csvhandler = cosem_db.CosemCSV(eval_results_csv_folder)
    col = db.access("evaluation", (db.training_version, db.gt_version))
    no_prediction = set()
    crop = db.get_crop_by_number(new_cropno)
    filter = {"crop": str(ref_cropno), "refined": False}
    if setup is not None:
        filter["setup"] = setup
    all_refs = [docu for docu in col.find(filter)]
    #with col.find(filter, batch_size=100, no_cursor_timeout=True) as cursor:
    for k, entry in enumerate(all_refs):
        print(".", end="", flush=True)
        setup = entry["setup"]
        iteration = entry["iteration"]
        s1 = "s1" in entry["raw_dataset"]
        

        try:
            pred_path = find_pred_path(setup, iteration, crop, s1, training_version=training_version)
        except FileNotFoundError:
             no_prediction.add({k: entry[k] for k in ["setup", "iteration", "label", "raw_dataset"]})
             continue
        # ds = zarr.open(pred_path, "r")[entry["label"]]
        # try: 
        #     if ds.attrs["mask_ds"] not in valid_masks:
        #         print()
        #         continue
        # except KeyError:
        #     print()
        #     continue
            
        test_query = {"path": pred_path, "dataset": entry["dataset"], "setup": setup, "iteration": iteration,
                    "label": entry["label"], "crop": str(new_cropno), "threshold": entry["threshold"],
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
                        db.get_crop_by_number(str(new_cropno)), entry["threshold"], metrics, metric_params, db,
                        csvhandler, True, False, gt_version=gt_version)
    return no_prediction    


def main():
    parser = argparse.ArgumentParser("Run evaluations for a newly added validation crop")
    parser.add_argument("--new_crop", type=int, default=111,
                        help="new crop on which to evaluate on")
    parser.add_argument("--ref_crop", type=int, default=111,
                        help="reference crop for which evaluations should be replicated on new crop")
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
    parser.add_argument("--save_missing_pred", type=str, default="missing_predictions.json", 
                        help="Filepath to which to save list of missing predicitons")
    parser.add_argument("--setup", type=str, default=None, help="Resetrict new evaluations to this setup")
    args = parser.parse_args()
    no_prediction = run_new_crop(args.new_crop, args.ref_crop, gt_version=args.gt_version, 
                                 training_version=args.training_version, tol_distance=args.tol_distance, 
                                 clip_distance=args.clip_distance, setup=args.setup)
    print("Missing predictions:")
    for nop in no_prediction:
        print(nop)
    with open(args.save_missing_pred, 'w') as f:
        json.dump(list(no_prediction), f)
    print("saved to", args.save_missing_pred)
    

if __name__ == "__main__":
    main()
