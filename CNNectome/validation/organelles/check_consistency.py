from gunpowder import Coordinate
import sys
import argparse
import CNNectome.validation.organelles.segmentation_metrics as segmentation_metrics
import CNNectome.utils.cosem_db as cosem_db
from CNNectome.utils.crop_utils import get_label_ids_by_category, check_label_in_crop

training_version = "v0003.2"


def check(setup, metric, metric_params, threshold, db):
    setup_dir = "/nrs/cosem/cosem/training/{training_version:}/{setup:}".format(training_version=training_version,
                                                                                setup=setup)
    sys.path.append(setup_dir)
    import unet_template
    labels = unet_template.labels
    is8 = unet_template.voxel_size_input == Coordinate((8,) * 3)
    label_to_cropnos = {}
    crops = db.get_all_validation_crops()
    if is8:
        raw_datasets = ["volumes/raw/s1", "volumes/subsampled/raw/0"]
    else:
        raw_datasets = ["volumes/raw"]
    for lbl in labels:
        for crop in crops:
            if check_label_in_crop(lbl, crop):
                try:
                    label_to_cropnos[lbl.labelname].append(crop["number"])
                except KeyError:
                    label_to_cropnos[lbl.labelname] = [crop["number"]]
    if len(label_to_cropnos) < len(labels):
        setup_labels = set([lbl.labelname for lbl in labels])
        crop_labels = set(label_to_cropnos.keys())
        for lblname in setup_labels - crop_labels:
            print("{0:} not in any crop".format(lblname))
    if len(label_to_cropnos) == 0:
        return True

    eval_col = db.access("evaluation", training_version)

    test_label, test_cropnos = next(iter(label_to_cropnos.items()))

    test_query = {"setup": setup,
                  "raw_dataset": raw_datasets[0],
                  "crop": test_cropnos[0],
                  "metric": metric,
                  "metric_params": metric_params,
                  "label": test_label,
                  "threshold": threshold,
                  "refined": False}

    num_test = eval_col.count_documents(test_query)
    iterations_test = set(x["iteration"] for x in eval_col.find(test_query, {"iteration": 1, "_id": 0}))
    will_return = True
    for lblname, cropnos in label_to_cropnos.items():
        for cropno in cropnos:
            for raw_ds in raw_datasets:
                query = {"setup": setup, "raw_dataset": raw_ds, "crop": cropno, "metric": metric,
                         "metric_params": metric_params, "label": lblname, "threshold": threshold, "refined": False}
                num = eval_col.count_documents(query)
                print("Query {0:} getting {1:}".format(query, num))
                iterations = set(x["iteration"] for x in eval_col.find(query, {"iteration": 1, "_id": 0}))
                # print(iterations)
                if iterations != iterations_test:
                    if len(iterations) > len(iterations_test):
                        print("Query {0:} missing {1:}".format(test_query, iterations-iterations_test))
                    else:
                        print("Query {0:} missing {1:}".format(query, iterations_test-iterations))
                    will_return = False

    print(sorted(list(iterations_test)))
    return will_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("setup", type=str)
    parser.add_argument("metric", type=str, choices=list(em.value for em in segmentation_metrics.EvaluationMetrics))
    parser.add_argument("--tol_distance", type=int, default=40)
    parser.add_argument("--clip_distance", type=int, default=200)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--db_password", type=str)
    parser.add_argument("--db_username", type=str)
    args = parser.parse_args()
    db = cosem_db.MongoCosemDB(args.db_username, args.db_password)
    metric_params = {"tol_distance": args.tol_distance,
                     "clip_distance": args.clip_distance}
    metric_params = segmentation_metrics.filter_params(metric_params, args.metric)
    print(check(args.setup, args.metric, metric_params, args.threshold, db))


if __name__ == "__main__":
    main()