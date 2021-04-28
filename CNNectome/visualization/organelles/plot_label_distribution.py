import json
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from CNNectome.utils.label import Label
import argparse

plt.rcParams["svg.fonttype"] = "none"


def get_data(filename):
    with open(filename, "r") as f:
        stats = json.load(f)
    return stats


def rearrange(labels, stats):
    pos_counts = []
    for label in labels:
        # label = Label("a", 4)
        pos_counts.append(0)
        for lid in label.labelid:
            pos_counts[-1] += stats["positives"][str(lid)]
        print(label.labelname, pos_counts[-1] / 5079502913)
    print(pos_counts)
    order = np.argsort(pos_counts)
    labelnames = [l.labelname for l in labels]
    # print(stats["negatives"]["3"]+stats["positives"]["3"])

    return (
        [pos_counts],
        labelnames,
        order,
        ["positives"],
        [(102 / 255.0, 190 / 255.0, 121 / 255.0)],
    )


def get_raw_stats(stats):
    label_ids = sorted([int(k) for k in stats["positives"].keys()])
    pos_counts = [stats["positives"][str(lid)] / 8 for lid in label_ids]
    neg_counts = [stats["negatives"][str(lid)] / 8 for lid in label_ids]
    sum_counts = [stats["sums"][str(lid)] / 8 for lid in label_ids]
    order = np.argsort(pos_counts)
    label_names = [
        "ECS",
        "PM",
        "mito mem",
        "mito lum",
        "mito DNA",
        "golgi mem",
        "golgi lum",
        "vesicle mem",
        "vesicle lum",
        "MVB mem",
        "MVB lum",
        "lysosome mem",
        "lysosome lum",
        "LD mem",
        "LD lum",
        "ER mem",
        "ER lum",
        "ERES mem",
        "ERES lum",
        "NE mem",
        "NE lum",
        "nuc. pore out",
        "nuc. pore in",
        "HChrom",
        "NHChrom",
        "EChrom",
        "NEChrom",
        "nucleoplasm",
        "nucleolus",
        "microtubules out",
        "centrosome",
        "distal app",
        "subdistal app",
        "ribos",
        "microtubules in",
        "nucleus generic",
    ]
    counts = [pos_counts, sum_counts]
    # counts = [pos_counts]
    # return counts, label_names, order, ["labeled positive",], [(255/255., 0/255., 127/255.)]
    return (
        counts,
        label_names,
        order,
        ["labeled positive", "labeled"],
        [(255 / 255.0, 0 / 255.0, 127 / 255.0), (45 / 255.0, 169 / 255.0, 72 / 255.0)],
    )


def plot_hist(counts, label_names, order=None, count_labels=None, colors=None, plotfile=None, transparent=True):
    # plt.style.use('dark_background')
    fs = 40
    # flist = matplotlib.font_manager.get_fontconfig_fonts()
    # x = matplotlib.font_manager.findSystemFonts(
    #     fontpaths="groups/saalfeld/home/heinrichl/fonts/webfonts/", fontext="ttf"
    # )
    # names = [
    #     matplotlib.font_manager.FontProperties(fname=fname).get_name()
    #     for fname in flist
    # ]
    plt.rcParams["font.family"] = "Nimbus Sans L"

    # plt.rcParams["font.sans-serif"] = "NimbusSansL"
    # print([(l, s) for l, s in zip(label_ids,sum_counts)])
    fig, ax = plt.subplots(figsize=(40, 15))

    x = np.arange(len(label_names))
    if order is None:
        order = list(range(len(label_names)))
    if count_labels is None:
        count_labels = [""] * len(counts)
    if colors is None:
        colors = [None] * len(counts)
    width = 0.85 / 2  # len(counts)
    # COLOR='white'
    # plt.rcParams['text.color'] = COLOR
    # plt.rcParams['axes.labelcolor'] = COLOR
    # plt.rcParams['xtick.color'] = COLOR
    # plt.rcParams['ytick.color'] = COLOR
    plt.ylim([5 * 10 ** 3, 0.8 * 10 ** 9])
    plt.xlim([-width, x[-1] + width * 2])

    shift = (len(counts) * width + (len(counts) - 1) * 0.01) / 2 - width / 2
    ax.set_xticks([xx + shift for xx in x])
    ax.yaxis.grid(linestyle=":", linewidth=2, c="k")
    ax.set_xticklabels(
        np.array(label_names)[order], rotation=60, ha="right", fontsize=fs
    )
    # ax.set_axisbelow(True)
    # ax.set_yticklabels()
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)
    plt.ylabel("# voxel", fontsize=fs)
    # ax.set_yticks([])
    plt.tick_params(axis="y", which="both", length=0)
    # print([(l, s) for l, s in zip(np.array(labelnames)[order], np.array(sum_counts)[order])])

    plt.semilogy()
    for k, (c, l, col) in enumerate(zip(counts, count_labels, colors)):
        if col is not None:
            col_int = col  # + (100/255., )
            rects = ax.bar(
                x + k * width + 0.01 * k,
                np.array(c)[order],
                width,
                label=l,
                color=col_int,
                edgecolor=col,
            )
        else:
            rects = ax.bar(x + k * width + 0.01 * k, np.array(c)[order], width, label=l)
            # (102/255.,190/255.,121/255.,100/255.), edgecolor = (102/255.,190/255.,121/255.))
    # rects2 = ax.bar(x + width/2, np.array(neg_counts)[order], width, label="labeled as negative", color=(219/255.,
    #                                                                                                  47/255.,
    #                                                                                      32/255., 100/255.),
    #              edgecolor = (219/255., 47/255., 32/255.))
    ax.legend(
        prop={"size": fs},
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=plt.gcf().transFigure,
        frameon=False,
    )
    # plt.gca().set(frame_on=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set(linewidth=2)
    plt.tight_layout(True)
    if plotfile is not None:
        plt.savefig(
            plotfile,
            transparent=transparent,
        )
    plt.show()


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--json", type=str,
                        help="json file with statistics, can be generated with `utils/compute_label_distribution.py`")
    parser.add_argument("--plotfile", type=str, help="Location in which to save figure.")
    parser.add_argument("--transparent", action="store_true", help="whether to save with transparent background")
    args = parser.parse_args()
    plot_hist(*get_raw_stats(get_data(args.json)), plotfile=args.plotfile, transparent=args.transparent)


if __name__ == "__main__":
    main()
