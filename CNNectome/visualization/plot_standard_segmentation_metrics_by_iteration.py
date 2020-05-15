import sys
sys.path = ["/groups/saalfeld/home/heinrichl/dev/CNNectome"] + sys.path
import zarr
from CNNectome.validation.organelles.standard_segmentation_metrics import EvaluationMetrics
import matplotlib.pyplot as plt
import numpy as np
from CNNectome.utils.label import Label


def read_individual_result(pred_path, label):
    n5file = zarr.open(pred_path, mode="r")
    it = n5file[label.labelname].attrs["iteration"]
    scores = {}
    for metric in EvaluationMetrics:
        scores[metric] = n5file[label.labelname].attrs[metric.name]["average"]
    return it, scores


def read_results(list_of_predictions, label):
    scores_by_it = {}
    for pred_path in list_of_predictions:
        it, scores = read_individual_result(pred_path, label)
        scores_by_it[it] = scores
    return scores_by_it

def best(argument):
    switcher = {
        EvaluationMetrics.dice:                np.nanargmax,
        EvaluationMetrics.jaccard:             np.nanargmax,
        EvaluationMetrics.hausdorff:           np.nanargmin,
        EvaluationMetrics.false_negatives:     np.nanargmin,
        EvaluationMetrics.false_positives:     np.nanargmin,
        EvaluationMetrics.adjusted_rand_index: np.nanargmax,
        EvaluationMetrics.voi:                 np.nanargmin
    }
    return switcher.get(argument)


def plot(label, predictions_to_compare, db_username, db_password):
    scores_by_it = read_results(predictions_to_compare, label)

    scores_by_it_sorted = sorted(scores_by_it.items())
    iterations = list(scores_by_it.keys())
    all_scores = list(scores_by_it.values())

    for metric in EvaluationMetrics:
        score = [s[metric] for s in all_scores]
        print(metric, score)
        line, = plt.plot(iterations, score , label = metric.name)
        opt = best(metric)(score)
        plt.plot(iterations[opt], score[opt], c = line.get_color(), alpha=0.5, marker='o')
    plt.legend()

def main():
    label = Label("er", (16, 17, 18, 19, 20, 21, 22, 23))
    prediction_paths = "/nrs/cosem/cosem/training/v0003.2/setup27.1/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm_it{iteration:}.n5"
    predictions_to_compare = (prediction_paths.format(iteration=it) for it in range(25000, 500001, 25000))

    db_username = "root"
    db_password = "root"
    plot(label, predictions_to_compare, db_username, db_password)
    plt.show()


if __name__ == "__main__":
    main()
