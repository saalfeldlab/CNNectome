import json
import os
import numpy as np
import matplotlib.table as tab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from CNNectome.utils import config_loader

iterations = list(range(30000, 150000, 2000))
colors = {
    "classic": (62 / 255.0, 150 / 255.0, 81 / 255.0),
    "lite": (57 / 255.0, 106 / 255.0, 177 / 255.0),
    "deluxe": (218 / 255.0, 124 / 255.0, 48 / 255.0),
}
linestyles = {
    "data2017-aligned": "-",
    "data2017-unaligned": "--",
    "data2016-aligned": "-.",
    "data2016-unaligned": ":",
}


def load_result(data_train, augmentation, data_eval, iteration, mode):
    result_json = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "data_and_augmentations",
        data_train,
        augmentation,
        "evaluation",
        str(iteration),
        data_eval,
        "cleft.{0:}.json".format(mode),
    )
    try:
        with open(result_json, "r") as f:
            resdict = json.load(f)
    except IOError:
        return None
    return resdict


def compute_cremi_score(samples, data_train, augmentation, data_eval, iteration, mode):
    result = load_result(data_train, augmentation, data_eval, iteration, mode)
    if result is None:
        return np.nan
    score = 0.0
    for s in samples:
        score += 0.5 * (
            result[s]["false positive distance"]["mean"]
            + result[s]["false negative distance"]["mean"]
        )
    score /= len(samples)
    return score


def compute_cremi_score_err(
    samples, data_train, augmentation, data_eval, iteration, mode
):
    result = load_result(data_train, augmentation, data_eval, iteration, mode)
    if result.isnan():
        return result
    err = 0.0
    for s in samples:
        err += 0.25 * (
            (result[s]["false positive distance"]["std"]) ** 2
            + (result[s]["false negative distance"]["std"]) ** 2
        )
    err = err ** (-0.5) / len(samples)
    return err


def compute_average_cremi_score(
    samples, data_train, augmentation, data_eval, iterations, mode
):
    res = []
    for i in iterations:
        res.append(
            compute_cremi_score(samples, data_train, augmentation, data_eval, i, mode)
        )
    avg_cremi_score = np.nanmean(res)
    std_cremi_score = np.nanstd(res)
    min_cremi_score = np.nanmin(res)
    print(len(res) - np.sum(np.isnan(res)))
    for k, r in enumerate(res):
        if not np.isnan(r):
            print(iterations[k])
    return avg_cremi_score, std_cremi_score, min_cremi_score


def compute_ratio(
    samples,
    data_train=None,
    augmentation=None,
    data_eval=None,
    comparisons=(("lite", "deluxe"), ("classic", "deluxe"), ("lite", "classic")),
):
    validation_ratios = dict()
    validation_min_ratios = dict()
    for comp in comparisons:
        validation_ratios[comp] = []

    comp_all = set([item for sublist in comparisons for item in sublist])
    res = dict()
    for c in comp_all:
        res[c] = []
    for i in iterations:
        for c in comp_all:
            if augmentation is None:
                aug, dt, de = c, data_train, data_eval
            elif data_train is None:
                aug, dt, de = augmentation, c, data_eval
            elif data_eval is None:
                aug, dt, de = augmentation, data_train, c
            res[c].append(compute_cremi_score(samples, dt, aug, de, i, "validation"))
        for comp in comparisons:
            validation_ratios[comp].append(res[comp[0]][-1] / res[comp[1]][-1])
    for comp in comparisons:
        validation_min_ratios[comp] = np.nanmin(res[comp[0]]) / np.nanmin(res[comp[1]])
    return validation_ratios, validation_min_ratios


def compute_average_ratio(
    samples,
    data_train=None,
    augmentation=None,
    data_eval=None,
    comparisons=(("lite", "deluxe"), ("classic", "deluxe"), ("lite", "classic")),
):
    validation_ratios, validation_min_ratios = compute_ratio(
        samples,
        data_train=data_train,
        augmentation=augmentation,
        data_eval=data_eval,
        comparisons=comparisons,
    )
    averages = dict()
    stds = dict()
    for comp in comparisons:
        averages[comp] = np.nanmean(validation_ratios[comp])
        stds[comp] = np.nanstd(validation_ratios[comp])

    # return validation_ratios, validation_min_ratios
    return averages, stds, validation_min_ratios


def plot_cremi_score_by_iteration(samples, data_train, augmentation, data_eval):
    training_cremi_scores = []
    validation_cremi_scores = []

    for i in iterations:
        training_cremi_scores.append(
            compute_cremi_score(
                samples, data_train, augmentation, data_eval, i, "training"
            )
        )
        validation_cremi_scores.append(
            compute_cremi_score(
                samples, data_train, augmentation, data_eval, i, "validation"
            )
        )
    minit_training = np.nanargmin(training_cremi_scores)
    minit_validation = np.nanargmin(validation_cremi_scores)
    plt.plot(
        iterations,
        training_cremi_scores,
        ls="--",
        c=colors[augmentation],
        label=augmentation + ", training",
        linewidth=0.5,
    )
    plt.plot(
        iterations[minit_training],
        training_cremi_scores[minit_training],
        c=colors[augmentation],
        marker="o",
        alpha=0.5,
    )
    plt.plot(
        iterations,
        validation_cremi_scores,
        ls="-",
        c=colors[augmentation],
        label=augmentation + ", validation",
    )
    plt.plot(
        iterations[minit_validation],
        validation_cremi_scores[minit_validation],
        c=colors[augmentation],
        marker="o",
    )
    plt.annotate(
        "{0:.2f}".format(validation_cremi_scores[minit_validation]),
        [iterations[minit_validation], validation_cremi_scores[minit_validation]],
        [4, -7],
        textcoords="offset points",
        color=colors[augmentation],
    )
    if "A" in samples and not "B" in samples and not "C" in samples:
        plt.ylim([0, 200])
    else:
        plt.ylim([0, 130])
    plt.legend()
    plt.xlabel("iterations")
    ylabel = "CREMI score on "
    for s in samples:
        ylabel += s
        if s != samples[-1]:
            ylabel += ", "
    plt.ylabel(ylabel)


def summary_plot(data_train, samples):
    trained_on = data_train
    samples = samples
    title = "trained on " + trained_on + ", evaluated on "
    if len(samples) > 1:
        title += "samples "
    else:
        title += "sample "
    for s in samples:
        title += s
        if s != samples[-1]:
            title += ", "
    fig = plt.figure(figsize=(30, 20))
    fig.suptitle(title)

    for sp, de in enumerate(
        [
            "data2017-aligned",
            "data2017-unaligned",
            "data2016-aligned",
            "data2016-unaligned",
        ]
    ):
        ax = plt.subplot(2, 2, sp + 1)
        ax.set_title("evaluated on " + de)
        for augmentation in ["deluxe", "classic", "lite"]:
            plot_cremi_score_by_iteration(samples, trained_on, augmentation, de)

    plotfile = os.path.join(
        config_loader.get_config()["synapses"]["training_setups_path"],
        "data_and_augmentations",
        trained_on,
        "".join(samples) + ".pdf",
    )

    plt.savefig(plotfile, transparent=True)
    # plt.show()


def all_summary_plots():

    for dt in [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]:
        for samples in [["A"], ["B"], ["C"], ["A", "B", "C"]]:
            summary_plot(dt, samples)


def ratio_summary_plots_augmentations():
    samples = ["A", "B", "C"]
    data_train = [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]
    data_eval = [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]
    symbols = {
        "data2017-aligned": "o",
        "data2017-unaligned": "^",
        "data2016-aligned": "s",
        "data2016-unaligned": "D",
    }
    colors = {
        ("lite", "deluxe"): "purple",
        ("classic", "deluxe"): "gold",
        ("lite", "classic"): "aqua",
    }

    fig = plt.figure(figsize=(30, 20))
    ax = plt.axes()
    d_tiny = 0.4
    d_small = 1.0
    d_large = 1.5

    plt.axhline(y=1.0, color="black", linestyle="--")
    x = d_small
    xt = []
    for de in data_eval:
        for k, dt in enumerate(data_train):
            avgs, stds, mins = compute_average_ratio(samples, dt, de)
            print(avgs, stds, mins)
            for l, comp in enumerate(avgs.keys()):
                plt.errorbar(
                    x,
                    avgs[comp],
                    yerr=stds[comp],
                    fmt=symbols[dt],
                    color=colors[comp],
                    ms=12,
                )
                plt.plot(
                    x,
                    mins[comp],
                    marker=symbols[dt],
                    color=colors[comp],
                    ms=7,
                    alpha=0.5,
                )
                x += d_tiny
                if (
                    k == np.ceil(len(data_train) / 2.0) - 1
                    and l == np.ceil(len(avgs) / 2.0) - 1
                ):
                    xt.append(x + ((len(avgs) + 1) % 2) * d_tiny / 2.0)

            x += d_small
        x += d_large
    plt.xticks(xt, ["eval on " + de for de in data_eval])
    ax.tick_params("x", labelsize="large", bottom=False)

    leg = []
    for comp, c in colors.items():
        p = mpatches.Patch(color=c, label=comp[0] + " / " + comp[1])
        leg.append(p)
    for dt in data_train:
        l = mlines.Line2D(
            [], [], color="black", marker=symbols[dt], ms=12, label="trained on " + dt
        )
        leg.append(l)
    l = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        ms=7,
        label="ratio of best performance",
        alpha=0.5,
    )
    leg.append(l)
    fig.legend(handles=leg)
    plt.ylabel("average ratio of CREMI scores")
    # plt.legend()
    plt.show()


def ratio_summary_plots_dt():
    samples = ["A", "B", "C"]
    # data_train= ['data2017-aligned', 'data2017-unaligned', 'data2016-aligned', 'data2016-unaligned']
    comparisons = (
        ("data2016-aligned", "data2017-aligned"),
        ("data2016-unaligned", "data2017-unaligned"),
        ("data2017-unaligned", "data2017-aligned"),
        ("data2016-unaligned", "data2016-aligned"),
    )
    augmentations = ["lite", "classic", "deluxe"]
    data_eval = [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]
    # symbols = {'data2017-aligned': 'o', 'data2017-unaligned': '^', 'data2016-aligned': 's', 'data2016-unaligned': 'D'}
    # colors = {('lite', 'deluxe'): 'purple', ('classic', 'deluxe'): 'gold', ('lite', 'classic'): 'aqua'}
    symbols = dict()
    markers = ["p", "v", "x", "*"]
    for c, s in zip(comparisons, markers):
        symbols[c] = s
    fig = plt.figure(figsize=(30, 20))
    ax = plt.axes()
    d_tiny = 0.4
    d_small = 1.0
    d_large = 1.5
    plt.axhline(y=1.0, color="black", linestyle="--")
    x = d_small
    xt = []
    for de in data_eval:
        for k, aug in enumerate(augmentations):
            avgs, stds, mins = compute_average_ratio(
                samples, data_eval=de, augmentation=aug, comparisons=comparisons
            )
            # ratios, mins = compute_average_ratio(samples, data_eval=de, augmentation=aug, comparisons=comparisons)
            print(avgs, stds, mins)
            for l, comp in enumerate(comparisons):
                plt.errorbar(
                    x,
                    avgs[comp],
                    yerr=stds[comp],
                    fmt=symbols[comp],
                    color=colors[aug],
                    ms=12,
                )
                # xcoor = np.linspace(x, x+0.5*d_tiny, len(ratios[comp]))
                # plt.plot(xcoor, ratios[comp], color=colors[aug], ms=12, marker=symbols[comp],ls='None')
                plt.plot(
                    x,
                    mins[comp],
                    marker=symbols[comp],
                    color=colors[aug],
                    ms=7,
                    alpha=0.5,
                )
                if (
                    k == np.ceil(len(augmentations) / 2.0) - 1
                    and l == np.ceil(len(avgs) / 2.0) - 1
                ):
                    xt.append(x + ((len(avgs) + 1) % 2) * d_tiny / 2.0)
                x += d_tiny

            x += d_small
        x += d_large
    plt.xticks(xt, ["eval on " + de for de in data_eval])
    ax.tick_params("x", labelsize="large", bottom=False)

    leg = []
    for aug, c in colors.items():
        p = mpatches.Patch(color=c, label=aug)
        leg.append(p)
    for c in comparisons:
        l = mlines.Line2D(
            [], [], color="black", marker=symbols[c], ms=12, label=c[0] + " / " + c[1]
        )
        leg.append(l)
    l = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        ms=7,
        label="ratio of best performance",
        alpha=0.5,
    )
    leg.append(l)
    fig.legend(handles=leg)
    plt.ylabel("average ratio of CREMI scores")
    # plt.legend()
    plt.show()


def absolute_summary_plots():
    samples = ["A", "B", "C"]
    augmentations = ["lite", "classic", "deluxe"]
    data_eval = [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]
    data_train = [
        "data2017-aligned",
        "data2017-unaligned",
        "data2016-aligned",
        "data2016-unaligned",
    ]
    # symbols = {'data2017-aligned': 'o', 'data2017-unaligned': '^', 'data2016-aligned': 's', 'data2016-unaligned': 'D'}
    # colors = {'data2017-aligned': 'b', 'data2017-unaligned': 'cyan', 'data2016-aligned': 'r', 'data2016-unaligned':
    #    'magenta'}
    abb = {
        "data2017-aligned": "17A",
        "data2017-unaligned": "17B",
        "data2016-aligned": "16A",
        "data2016-unaligned": "B",
    }
    plt.rcParams["font.sans-serif"] = ["Lucida Grande"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams.update({"font.size": 22})
    print(list(plt.rcParams.keys()))  # , 'font.name': 'Comic Sans MS'})
    fig = plt.figure(figsize=(30, 20))

    ax = plt.axes()
    for p in ax.spines.keys():
        ax.spines[p].set_visible(False)
    ax.grid(which="major", axis="y", linestyle=":", linewidth=2)
    d_tiny = 0.4
    d_small = 1.0
    d_large = 1.5

    x = d_small
    xt = []

    for de in data_eval:
        x += d_large
        for k, aug in enumerate(augmentations):
            for l, dt in enumerate(data_train):
                avg_cremi_score, std_cremi_score, min_cremi_score = compute_average_cremi_score(
                    samples, dt, aug, de, iterations, "validation"
                )
                plt.errorbar(
                    x,
                    avg_cremi_score,
                    yerr=std_cremi_score,
                    fmt="o",
                    color=colors[aug],
                    ms=12,
                    linewidth=2,
                )
                plt.plot(x, min_cremi_score, marker="o", ms=4, color=colors[aug])
                # c = tab.Cell((x, -1),0.5, 0.5, text=abb[de],loc='bottom')
                # c.set_figure(fig)
                # c = tab.Cell((x, -2),0.5,0.5, text=abb[dt], loc='bottom')
                # c.set_figure(fig)
                if (
                    k == np.ceil(len(augmentations) / 2.0) - 1
                    and l == np.ceil(len(data_train) / 2.0) - 1
                ):
                    xt.append(x + ((len(data_train) + 1) % 2) * d_tiny / 2.0)
                x += d_tiny
            x += d_small
    plt.xticks(xt, ["eval on " + de for de in data_eval])
    ax.tick_params("x", bottom=False)
    ax.tick_params("y", left=False)
    leg = []
    for aug in augmentations:
        l = mlines.Line2D([], [], color=colors[aug], marker="o", ms=12, label=aug)
        leg.append(l)
    l = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        ms=4,
        label="best CREMI score",
        linestyle="None",
    )
    leg.append(l)
    fig.legend(handles=leg)
    plt.ylabel("CREMI score")
    plt.show()


if __name__ == "__main__":
    absolute_summary_plots()
    # ratio_summary_plots_dt()
    # all_summary_plots()
