import json
import os
import numpy as np
import matplotlib.table as tab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

iterations = range(30000, 210000, 10000)
colors = {
    "classic": (62 / 255.0, 150 / 255.0, 81 / 255.0),
    "lite": (57 / 255.0, 106 / 255.0, 177 / 255.0),
    "deluxe": (218 / 255.0, 124 / 255.0, 48 / 255.0),
    "data2016-aligned": (218 / 255.0, 124 / 255.0, 48 / 255.0),
    "data2016-unaligned": (57 / 255.0, 106 / 255.0, 177 / 255.0),
    "data2017-unaligned": (62 / 255.0, 150 / 255.0, 81 / 255.0),
    "data2017-aligned": (204 / 255.0, 37 / 255.0, 41 / 255.0),
}
linestyles = {
    "data2017-aligned": "-",
    "data2017-unaligned": "--",
    "data2016-aligned": "-.",
    "data2016-unaligned": ":",
}


def load_result(data_train, augmentation, data_eval, iteration, mode):
    result_json = os.path.join(
        "/nrs/saalfeld/heinrichl/synapses/data_and_augmentations",
        data_train,
        augmentation,
        "evaluation",
        str(iteration),
        data_eval,
        "partners.{0:}.json".format(mode),
    )
    try:
        with open(result_json, "r") as f:
            resdict = json.load(f)
    except IOError:
        return None
    return resdict


def compute_cremi_score(
    samples, data_train, augmentation, data_eval, iteration, mode, metric="fscore"
):
    result = load_result(data_train, augmentation, data_eval, iteration, mode)
    if result is None:
        return 1.0
    score = 0.0
    for s in samples:
        if result[s][metric] is not None:
            score += result[s][metric]
        else:
            score += 0.0
    score /= len(samples)
    return 1.0 - score


def plot_cremi_score_by_iteration(
    samples,
    data_train,
    augmentation,
    data_eval,
    metric="fscore",
    color=(62 / 255.0, 150 / 255.0, 81 / 255.0),
):
    training_cremi_scores = []
    validation_cremi_scores = []

    for i in iterations:
        training_cremi_scores.append(
            compute_cremi_score(
                samples,
                data_train,
                augmentation,
                data_eval,
                i,
                "training",
                metric=metric,
            )
        )
        validation_cremi_scores.append(
            compute_cremi_score(
                samples,
                data_train,
                augmentation,
                data_eval,
                i,
                "validation",
                metric=metric,
            )
        )
    minit_training = np.nanargmin(training_cremi_scores)
    minit_validation = np.nanargmin(validation_cremi_scores)
    plt.plot(
        iterations,
        training_cremi_scores,
        ls="--",
        c=color,
        label=data_eval + ", training",
        linewidth=0.5,
    )
    plt.plot(
        iterations[minit_training],
        training_cremi_scores[minit_training],
        c=color,
        marker="o",
        alpha=0.5,
    )
    plt.plot(
        iterations,
        validation_cremi_scores,
        ls="-",
        c=color,
        label=data_eval + ", validation",
    )
    plt.plot(
        iterations[minit_validation],
        validation_cremi_scores[minit_validation],
        c=color,
        marker="o",
    )
    plt.annotate(
        "{0:.2f}".format(validation_cremi_scores[minit_validation]),
        [iterations[minit_validation], validation_cremi_scores[minit_validation]],
        [4, -7],
        textcoords="offset points",
        color=color,
    )

    plt.ylim([0.1, 1.0])
    plt.xlim([20000, 210000])

    plt.legend()
    plt.xlabel("iterations")
    ylabel = "CREMI score on "
    for s in samples:
        ylabel += s
        if s != samples[-1]:
            ylabel += ", "
    plt.ylabel(ylabel)


if __name__ == "__main__":
    samples = ["A", "B", "C"]
    data_train = ["data2017-aligned", "data2017-unaligned"]
    data_eval = ["data2017-aligned", "data2017-unaligned"]
    for k, dt in enumerate(data_train):
        plt.subplot("".join(("2", str(len(data_train)), str(k + 1))))
        plt.title("train on " + dt)
        for de in data_eval:
            plot_cremi_score_by_iteration(samples, dt, "deluxe", de, color=colors[de])
    data_train = ["data2016-aligned", "data2016-unaligned"]
    data_eval = ["data2016-aligned", "data2016-unaligned"]
    for l, dt in enumerate(data_train):
        plt.subplot("".join(("2", str(len(data_train)), str(k + l + 2))))
        plt.title("train on " + dt)
        for de in data_eval:
            plot_cremi_score_by_iteration(samples, dt, "deluxe", de, color=colors[de])

    plt.show()
