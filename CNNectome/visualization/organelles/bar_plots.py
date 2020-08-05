from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.validation.organelles.segmentation_metrics import *
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

host = "cosem.int.janelia.org:27017"
gt_version = 'v0003'
training_version = 'v0003.2'
colors = [(45/255., 169/255., 72/255.), "magenta", "orange"]


def plot_double_bars(labels, left, right, metric, left_label, right_label, cropnames, name="s1_vs_sub"):
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    # men_means = [20, 34, 30, 35, 27]
    # women_means = [25, 32, 34, 20, 25]
    sns.set_style('darkgrid')
    fs = 30

    plt.rcParams["font.family"] = "Nimbus Sans L"
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})

    x = np.arange(len(list(labels.values())[0]))  # the label locations
    print(x)
    width = 0.35/2.  # the width of the bars

    fig, ax = plt.subplots(figsize=(20,15))

    kwargs = [[{"color": colors[0], "edgecolor": colors[0], "label":left_label},
               {"color": colors[1], "edgecolor": colors[1], "label":right_label}],
              [{"facecolor": (0, 0, 0, 0), "hatch": "xx", "edgecolor": colors[0]},
               {"facecolor": (0, 0, 0, 0), "hatch": "xx", "edgecolor": colors[1]}]]
    for cno, shift, kw in zip(left.keys(), [-width/2, width/2], kwargs):
        print(left[cno])
        rects1 = ax.bar(x -  width + shift , left[cno], width, **kw[0])
        rects2 = ax.bar(x + width  + shift, right[cno], width, **kw[1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if "distance" in metric:
        ylabel = metric + " [nm]"
    else:
        ylabel = metric
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels.values())[0], rotation=60, ha="right", fontsize=fs)
    ax.tick_params(axis="y", which="major", labelsize=fs)
    leg = ax.legend(fontsize=fs, prop={"size": fs}, loc=1)
    ax.add_artist(leg)
    patch1 = mpatches.Patch(color="w", label=cropnames[list(left.keys())[0]])
    patch2 = mpatches.Patch(facecolor=(0,0,0,1), edgecolor="w", hatch="xx", label=cropnames[list(left.keys())[1]])
    leg2 = plt.legend(handles=[patch1, patch2], fontsize=fs, prop={"size": fs}, loc=2)
    for text in leg.get_texts():
        plt.setp(text, color="w")
    for text in leg2.get_texts():
        plt.setp(text, color="w")
    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/figures/COSEM/"+name+"_" + metric+'.png'), transparent=True)
    plt.show()


def plot_triple_bars(labels, left, middle, right, metric, left_label, middle_label, right_label, cropnames,
                     name="s1_vs_sub"):
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    # men_means = [20, 34, 30, 35, 27]
    # women_means = [25, 32, 34, 20, 25]
    sns.set_style('darkgrid')
    fs = 25

    plt.rcParams["font.family"] = "Nimbus Sans L"
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})

    x = np.arange(len(list(labels.values())[0]))  # the label locations
    width = 0.13  # the width of the bars

    fig, ax = plt.subplots(figsize=(30,15))

    kwargs = [[{"facecolor": colors[0], "edgecolor": colors[0], "label": left_label},
               {"facecolor": colors[1], "edgecolor": colors[1], "label": middle_label},
               {"facecolor": colors[2], "edgecolor": colors[2], "label": right_label}],
              [{"facecolor": (0, 0, 0, 1), "hatch": "xx", "edgecolor": colors[0]},
               {"facecolor": (0, 0, 0, 1), "hatch": "xx", "edgecolor": colors[1]},
               {"facecolor": (0, 0, 0, 1), "hatch": "xx", "edgecolor": colors[2]}]
              ]
    for cno, shift, kw in zip(left.keys(), [-width/2, width/2], kwargs):
        rects1 = ax.bar(x - 2 * width + shift, left[cno], width, **kw[0])
        rects2 = ax.bar(x + shift, middle[cno], width, **kw[1])
        rects3 = ax.bar(x + 2 * width + shift, right[cno], width, **kw[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if "distance" in metric:
        ylabel = metric + " [nm]"
    else:
        ylabel = metric
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels.values())[0], rotation=60, ha="right", fontsize=fs)
    ax.tick_params(axis="y", which="major", labelsize=fs)
    leg = ax.legend(fontsize=fs, prop={"size": 20}, loc=1)
    ax.add_artist(leg)
    patch1 = mpatches.Patch(color="w", label=cropnames[list(left.keys())[0]])
    patch2 = mpatches.Patch(facecolor=(0, 0, 0, 1), edgecolor="w", hatch="xx", label=cropnames[list(left.keys())[1]])
    leg2 = plt.legend(handles=[patch1, patch2], fontsize=fs, prop={"size": 20}, loc=2)
    for text in leg.get_texts():
        plt.setp(text, color="w")
    for text in leg2.get_texts():
        plt.setp(text, color="w")
    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/figures/COSEM/"+name+"_" + metric+'.png'), transparent=True)
    plt.show()


def plot_single_bars(labels, scores, metric):

    sns.set_style('darkgrid')
    fs = 50

    plt.rcParams["font.family"] = "Nimbus Sans L"
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",})
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(30, 15))
    color = colors[0]
    kwargs = ({"color": color}, {"color": (0, 0, 0, 1), "hatch": "xxx"})
    for (cropname, values), shift, kw in zip(scores.items(), [-width/2, width/2], kwargs):
        rects = ax.bar(x + shift, values, width, label=cropname, edgecolor=color, **kw)

    if "distance" in metric:
        ylabel = metric + " [nm]"
    else:
        ylabel = metric
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=fs)
    ax.tick_params(axis="y", which="major", labelsize=fs)
    leg = ax.legend(fontsize=fs, prop={"size": 30})
    for text in leg.get_texts():
        plt.setp(text, color="w")
    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/figures/COSEM/best_per_label_" + metric+'.png'), transparent=True)
    plt.show()


def s1_vs_sub(db_username, db_password, metric, cropno, cropnames, tol_distance=40, clip_distance=200, threshold=127):
    setups = ["setup26.1", "setup28", "setup32", "setup46"]
    labels = [("mito", "mito_membrane"), ("er", "er_membrane"), ("microtubules", "microtubules_out"),
              ("ecs", "plasma_membrane")]

    db = cosem_db.MongoCosemDB(db_username, db_password, host=host, gt_version=gt_version,
                               training_version=training_version)
    values = {}

    for cno in cropno:
        values[cno] = []
        for setup, label in zip(setups, labels):
            for lbl in label:
                best_s1 = analyze_evals.best_result(db, lbl, [setup], cno, metric, raw_ds="volumes/raw/s1",
                                                    tol_distance=tol_distance, clip_distance=clip_distance,
                                                    threshold=threshold)
                best_sub = analyze_evals.best_result(db, lbl, [setup], cno, metric, raw_ds="volumes/subsampled/raw/0",
                                                     tol_distance=tol_distance, clip_distance=clip_distance,
                                                     threshold=threshold)
                values[cno].append((setup, l, best_s1["value"], best_sub["value"]))

    labels = {}
    sub = {}
    s1 = {}
    names = dict((k,v) for k,v in zip(cropno, cropnames))
    for cno in cropno:

        labels[cno] = [v[1] for v in values[cno]]
        sub[cno] = [v[3] for v in values[cno]]
        s1[cno] = [v[2] for v in values[cno]]

    plot_double_bars(labels, sub, s1, metric, "subsampled", "average", names)


def plot_4nm_vs_8nm(db_username, db_password, metric, cropno, cropnames, tol_distance=40, clip_distance=200,
                    threshold=127):
    setups_4nm = ["setup25", "setup27.1", "setup31", "setup45"]
    setups_8nm = ["setup26.1", "setup28", "setup32", "setup46"]
    labels = ["mito", "mito_membrane", "er", "microtubules", "microtubules_out",
              "ecs", "plasma_membrane"]
    metric_params = dict()
    metric_params["clip_distance"] = clip_distance
    metric_params["tol_distance"] = tol_distance
    specific_params = filter_params(metric_params, metric)
    db = cosem_db.MongoCosemDB(db_username, db_password, host=host, gt_version=gt_version,
                               training_version=training_version)
    values = {}
    for cno in cropno:
        values[cno] = []

        for lbl in labels:
            best_4nm = best_result(db, lbl, setups_4nm, cropno, metric, tol_distance=tol_distance,
                                       clip_distance=clip_distance, threshold=threshold)
            best_8nm = best_result(db, lbl, setups_8nm, cropno, metric, raw_ds="volumes/subsampled/raw/0",
                                       tol_distance=tol_distance, clip_distance=clip_distance, threshold=threshold)
            values[cno].append((lbl, best_4nm["value"], best_8nm["value"]))
    labels = {}
    v_4nm = {}
    v_8nm = {}
    names = dict((k, v) for k, v in zip(cropno, cropnames))
    for cno in cropno:
        labels[cno] = [v[0] for v in values[cno]]
        v_4nm[cno] = [v[1] for v in values[cno]]
        v_8nm[cno] = [v[2] for v in values[cno]]

    print("4nm", v_4nm)
    print("8nm", v_8nm)
    plot_double_bars(labels, v_4nm, v_8nm, metric, "4nm input", "8nm input", names)


def plot_all_vs_common_vs_single(db_username, db_password, metric, cropno, cropnames, tol_distance=40,
                                 clip_distance=200, threshold=127):
    setups_all = ["setup01", "setup01", "setup01", "setup01", "setup01"]
    setups_common = ["setup03", "setup03", "setup03", "setup03", "setup03"]
    setups_single = ["setup25", "setup27.1", "setup31", "setup35", "setup45"]
    labels = ["mito", "mito_membrane", "er",  "er_membrane", "microtubules", "nucleus", "ecs", "plasma_membrane"]
    db = cosem_db.MongoCosemDB(db_username, db_password, host=host, gt_version=gt_version,
                               training_version=training_version)
    labels = {}
    v_all = {}
    v_common = {}
    v_single = {}
    for cno in cropno:
        v_all[cno] = []
        v_common[cno] = []
        v_single[cno] = []
        labels[cno] = []
        for lbl in labels:
            best_all = analyze_evals.best_result(db, lbl, setups_all, cno, metric, tol_distance=tol_distance,
                                                 clip_distance=clip_distance, threshold=threshold)
            best_common = analyze_evals.best_result(db, lbl, setups_common, cno, metric, tol_distance=tol_distance,
                                                 clip_distance=clip_distance, threshold=threshold)
            best_single = analyze_evals.best_result(db, lbl, setups_single, cno, metric, tol_distance=tol_distance,
                                                 clip_distance=clip_distance, threshold=threshold)
            v_all[cno].append(best_all["value"])
            v_common[cno].append(best_common["value"])
            v_single[cno].append(best_single["value"])
            labels[cno].append(lbl)

    plot_triple_bars(labels, v_all, v_common, v_single, metric, "all classes", "common classes", "single/few classes",
                     names, name="all_vs_common_vs_single")


def best_per_label(db_username, db_password, metric, cropno, cropnames, tol_distance=40, clip_distance=200):
    labels = ["ecs", "plasma_membrane", "mito", "mito_membrane", "vesicle", "er",
              "er_membrane", "ERES", "microtubules", "nucleus"]
    setups = ["setup01", "setup03", "setup25", "setup27.1", "setup31", "setup35", "setup45", "setup47", "setup49"]
    metric_params = dict()
    metric_params["clip_distance"] = clip_distance
    metric_params["tol_distance"] = tol_distance
    db = cosem_db.MongoCosemDB(db_username, db_password, host=host, gt_version=gt_version,
                               training_version=training_version)
    bests = {}
    for cno, cropname in zip(cropno, cropnames):
        bests[cropname] = []
    for cno, cropname in zip(cropno, cropnames):
        for lbl in labels:
            best_setup = analyze_evals.best_result(db, lbl, setups, cropno, metric, tol_distance=tol_distance,
                                                   clip_distance=clip_distance)
            bests[cropname].append(best_setup["value"])
    plot_single_bars(labels, bests, metric)


def main():
    parser = argparse.ArgumentParser("Plot bar plots")
    parser.add_argument("plot", type=str, choices=["s1_vs_sub", "best_per_label", "4nm_vs_8nm",
                                                   "all_vs_common_vs_single"])
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")
    parser.add_argument("--metric", type=str, default=None, help="metric to evaluate",
                        choices=list(em.value for em in EvaluationMetrics))
    parser.add_argument("--cropno", type=int, nargs="+")
    parser.add_argument("--cropnames", type=str, nargs="+")
    args = parser.parse_args()
    if args.plot == "s1_vs_sub":
        s1_vs_sub(args.db_username, args.db_password, args.metric, args.cropno, args.cropnames)
    elif args.plot == "best_per_label":
        best_per_label(args.db_username, args.db_password, args.metric, args.cropno, args.cropnames)
    elif args.plot == "4nm_vs_8nm":
        plot_4nm_vs_8nm(args.db_username, args.db_password, args.metric, args.cropno, args.cropnames)
    elif args.plot == "all_vs_common_vs_single":
        plot_all_vs_common_vs_single(args.db_username, args.db_password, args.metric, args.cropno, args.cropnames)


if __name__ == "__main__":
    main()
