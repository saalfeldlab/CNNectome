from CNNectome.validation.organelles.run_evaluation import *
from CNNectome.validation.organelles.segmentation_metrics import *
from CNNectome.utils.hierarchy import short_names
from CNNectome.validation.organelles.comparisons import compare
from CNNectome.utils import config_loader, cosem_db
import csv
import collections
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from operator import itemgetter
import os
from typing import Dict, List, Optional, Sequence, Union

fs = 10
ratio = 1.6
fig_width_per_label = 0.32
fig_height = 2.55
sym_val = 'v'
sym_test = 'o'
sym_manual = 's'


def sort_generic(labels: Sequence[str],
                 sorting_metric: str = "dice",
                 from_csv: bool = False,
                 db: Optional[cosem_db.MongoCosemDB] = None,
                 training_version: str = "v0003.2",
                 tol_distance: int = 40,
                 clip_distance: int = 200) -> List[str]:
    """
    Sort labels by their average test score.

    Args:
        labels: List of labels to be sorted.
        sorting_metric: Metric to use for sorting.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.

    Returns:
        List of sorted labels.
    """
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons",
                                "best-4nm_across-setups_test_{0:}.csv".format(sorting_metric))
        reader = csv.DictReader(open(csv_file, "r"))
        results = []
        for row in reader:
            results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        results = compare("best-4nm", db, [sorting_metric], tol_distance=tol_distance,
                          clip_distance=clip_distance, mode="across_setups", test=True)

    for result in results:
        result["label"] = short_names[result["label"]]
    avgs = dict()
    for label in labels:
        avgs[label] = []
    for result in results:
        if result["label"] in labels:
            if result["value"] == "":
                if sorting_metric == "mean_false_distance":
                    avgs[result["label"]].append(np.nan)
                elif sorting_metric == "dice":
                    avgs[result["label"]].append(0)
            else:
                avgs[result["label"]].append(float(result["value"]))
    for k, v in avgs.items():
        avgs[k] = np.nanmean(v)
    avgs_tuples = [(v, k) for k, v in avgs.items()]
    labels = [l for _, l in sorted(avgs_tuples)]
    labels = labels[::sorting(sorting_metric)]
    return labels


def plot_generalization(metric: str, label: str,
                        from_csv: bool = False,
                        db: Optional[cosem_db.MongoCosemDB] = None,
                        training_version: str = "v0003.2",
                        tol_distance: int = 40,
                        clip_distance: int = 200,
                        filetype: str = "svg",
                        transparent: bool = False,
                        save: bool = False,
                        to_csv: Optional[str] = None,
                        ) -> None:
    """
    Plot generalization metric.

    Args:
        metric: Metric to use for comparison.
        label: Label for which to plot generalization matrix.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figures.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons", "generalization_{0:}.csv".format(metric))
        reader = csv.DictReader(open(csv_file, "r"))
        all_results = []
        for row in reader:
            all_results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        all_results = compare("generalization", db, [metric], tol_distance=tol_distance,
                              clip_distance=clip_distance)
    remove_entries = []
    for k, result in enumerate(all_results):
        if result["label"] != label:
            remove_entries.append(k)
    all_results = [r for k, r in enumerate(all_results) if k not in remove_entries]

    # crops = collections.OrderedDict([("jrc_hela-2", "113")
    #                                 , ("jrc_jurkat-1", "112"),("jrc_hela-3", "111"), ("jrc_macrophage-2", "110")])
    # setups = collections.OrderedDict(
    #     [("jrc_hela-2", "setup61"),  ("jrc_jurkat-1", "setup64"), ("jrc_hela-3", "setup63"),
    #      ("jrc_macrophage-2", "setup62"), ("combined", "setup03")])
    crops = collections.OrderedDict([("jrc_hela-2", "113"),
                                     ("jrc_hela-3", "111"),
                                     ("jrc_jurkat-1", "112"),
                                     ("jrc_macrophage-2", "110"),
                                     ])
    setups = collections.OrderedDict([("jrc_hela-2", "setup61"),
                                      ("jrc_hela-3", "setup63"),
                                      ("jrc_jurkat-1", "setup64"),
                                      ("jrc_macrophage-2", "setup62"),
                                      ("combined", "setup03")])
    values = []
    result_arr = np.zeros((4, 5))
    fig, ax = plt.subplots()
    csv_results = []
    for k, (crop_dataset, c) in enumerate(reversed(crops.items())):
        for j, (setup_dataset, s) in enumerate(setups.items()):
            for r in all_results:
                if r["crop"] == c and r["setup"] == s:
                    result_arr[k][j] = float(r["value"])
                    csv_entry = {"evaluation_dataset": crop_dataset,
                                 "training_dataset": setup_dataset,
                                 "value": float(r["value"])
                    }
                    csv_results.append(csv_entry)
                    #print(float(r["value"]))
                    break
    if to_csv is not None:
        csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                "figures", to_csv)
        fields = ["evaluation_dataset", "training_dataset", "value"]
        csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(csv_results)
    vmin = 0
    if label == "er":
        cm = "Greens"
    elif label == "mito":
        cm = "Oranges"
    elif label == "nucleus":
        cm = "Purples"
    elif label == "plasma_membrane":
        #cm = "Greys"
        colors = [(.8,.8,.8), (.1,.1,.1)]
        n_bins = 255
        cmap_name = "gray_to_black"
        cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    else:
        cm = "gray"
        vmin = 0

    im = ax.imshow(result_arr, vmin=vmin, vmax=1, cmap=cm)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(display_name(metric), rotation=-90, va="bottom")
    plt.yticks(list(range(0, 4)), list(reversed(crops.keys())), fontsize=fs)
    plt.xticks(list(range(0, 5)), list(setups.keys()),  rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    ax.set_aspect("equal")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel("validation block")
    plt.xlabel("training dataset")
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "generalization_{0:}_{1:}_{2:}.{3:}".format(metric, label, transparent, filetype)),
            transparent=transparent)
        plt.show()


def plot_s1_vs_sub(metric: str,
                   from_csv: bool = False,
                   db: Optional[cosem_db.MongoCosemDB] = None,
                   training_version: str = "v0003.2",
                   tol_distance: int = 40,
                   clip_distance: int = 200,
                   filetype: str = "svg",
                   transparent: bool = False,
                   save: bool = False) -> None:
    """
    Plot the comparison of predicting on simulated 8nm data by randomly subsampling or averaging.

    Args:
        metric: Metric to use for comparison.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
    """

    fig_width_per_label = 0.279
    fig_height = 2.10505
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons", "s1-vs-sub_per-setup_test_{0:}.csv".format(metric))
        reader = csv.DictReader(open(csv_file, "r"))
        all_results = []
        for row in reader:
            all_results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        all_results = compare("s1-vs-sub", db, [metric], tol_distance=tol_distance, clip_distance=clip_distance,
                              mode="per_setup", test=True)
    labels = []
    for result in all_results:
        result["label_s1"] = short_names[result["label_s1"]]
        result["label_subsampled"] = short_names[result["label_subsampled"]]
        labels.append(result["label_s1"])
    labels = list(set(labels))
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for result in all_results:
    #     avgs[result["label_4nm"]].append(float(result["value_4nm"]))
    #     avgs[result["label_8nm"]].append(float(result["value_8nm"]))
    # for k, v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    # avgs_tuples = [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting(metric)]
    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)
    mids = list(range(1, 1 + len(labels)))
    x_values = []
    y_values = []
    colors = []
    off = 0.25
    color_dict = {"subsampled": "#1b9e77", "averaged": "#d95f02"}
    for m, label in enumerate(labels):
        mid = mids[m]
        for result in all_results:
            if result["label_subsampled"] == label:
                x_values.append(mid - off)
                y_values.append(float(result["value_subsampled"]))
                colors.append(color_dict["subsampled"])
                x_values.append(mid + off)
                y_values.append(float(result["value_s1"]))
                colors.append(color_dict["averaged"])
    vals = {}
    cols = {}
    for xx, yy, c in zip(x_values, y_values, colors):
        cols[xx] = c
        try:
            vals[xx].append(yy)
        except KeyError:
            vals[xx] = [yy]

    plt.figure(figsize=(fig_width_per_label * len(labels), fig_height))
    ax = plt.gca()
    ax.set_axisbelow(True)
    for mid in mids[:-1]:
        ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
    plt.grid(axis="y", color="gray", linestyle=(0, (1, 5)), linewidth=.5)
    for xxk, yyv in vals.items():
        plt.plot([xxk] * len(yyv), yyv, color=cols[xxk], linewidth=2.5, alpha=.5)
    ax.scatter(x_values, y_values, c=colors, s=fs * ratio, marker=sym_test)
    for cat, col in color_dict.items():
        ax.scatter([], [], c=col, label=cat, s=fs * ratio, marker=sym_test)

    ax.set_xticks(mids)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which='both', bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis="y", pad=15)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    if metric == "mean_false_distance":
        ax.set_yscale('log')
    plt.legend(frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025)
    plt.xlim([0.5, max(mids) + 0.5])
    if metric == "dice":
        plt.ylim([-.02, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.ylabel(display_name(metric))
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "s1-vs-ss_{0:}_{1:}.{2:}".format(metric, transparent, filetype)), transparent=transparent)
    plt.show()


def plot_4nm_vs_8nm(metric: str,
                    from_csv: bool = False,
                    db: Optional[cosem_db.MongoCosemDB] = None,
                    training_version: str = "v0003.2",
                    tol_distance: int = 40,
                    clip_distance: int = 200,
                    filetype: str = "svg",
                    transparent: bool = False,
                    save: bool = False, 
                    to_csv: Optional[str] = None) -> None:
    """
    Plot the comparison of predicting on simulated 8nm data by randomly subsampling or original 4nm data.

    Args:
        metric: Metric to use for comparison.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.

    Returns:
        None.
    """
    fig_width_per_label = 0.279
    fig_height = 2.10505
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons", "4nm-vs-8nm_across-setups_test_{0:}.csv".format(metric))
        reader = csv.DictReader(open(csv_file, "r"))
        all_results = []
        for row in reader:
            all_results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        all_results = compare("4nm-vs-8nm", db, [metric], tol_distance=tol_distance, clip_distance=clip_distance,
                              mode="across_setups", test=True)
    labels = []
    for result in all_results:
        result["label_4nm"] = short_names[result["label_4nm"]]
        result["label_8nm"] = short_names[result["label_8nm"]]
        labels.append(result["label_4nm"])
    labels = list(set(labels))
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for result in all_results:
    #     avgs[result["label_4nm"]].append(float(result["value_4nm"]))
    #     avgs[result["label_8nm"]].append(float(result["value_8nm"]))
    # for k, v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    # avgs_tuples = [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting(metric)]

    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)
    mids = list(range(1, 1 + len(labels)))
    x_values = []
    y_values = []
    colors = []
    off = 0.25
    color_dict = {"4nm network": "#1b9e77", "8nm network": "#d95f02"}
    crop_ds = {"111": "jrc_hela-3",
               "113": "jrc_hela-2",
               "110": "jrc_macrophage-2",
               "112": "jrc_jurkat-1"}
    csv_results = []
    for m, label in enumerate(labels):
        mid = mids[m]
        for result in all_results:
            if result["label_4nm"] == label:
                csv_entry = {"label": label, "dataset": crop_ds[result["crop_4nm"]]}
                x_values.append(mid - off)
                y_values.append(float(result["value_4nm"]))
                csv_entry["4nm_network"] = y_values[-1]
                colors.append(color_dict["4nm network"])
                x_values.append(mid + off)
                y_values.append(float(result["value_8nm"]))
                csv_entry["8nm_network"] = y_values[-1]
                colors.append(color_dict["8nm network"])
                csv_results.append(csv_entry)
    if to_csv is not None:
        csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                "figures", to_csv)
        fields = ["label", "dataset", "4nm_network", "8nm_network"]
        csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(csv_results)
    vals = {}
    cols = {}
    for xx, yy, c in zip(x_values, y_values, colors):
        cols[xx] = c
        try:
            vals[xx].append(yy)
        except KeyError:
            vals[xx] = [yy]

    plt.figure(figsize=(fig_width_per_label * len(labels), fig_height))
    ax = plt.gca()
    ax.set_axisbelow(True)
    for mid in mids[:-1]:
        ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
    plt.grid(axis="y", color="gray", linestyle=(0, (1, 5)), linewidth=.5)
    for xxk, yyv in vals.items():
        plt.plot([xxk] * len(yyv), yyv, color=cols[xxk], linewidth=2.5, alpha=.5)
    ax.scatter(x_values, y_values, c=colors, s=fs * ratio, marker=sym_test)
    for cat, col in color_dict.items():
        ax.scatter([], [], c=col, label=cat, s=fs * ratio, marker=sym_test)

    ax.set_xticks(mids)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which='both', bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis="y", pad=15)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    if metric == "mean_false_distance":
        ax.set_yscale('log')
    plt.legend(frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025)
    plt.xlim([0.5, max(mids) + 0.5])
    if metric == "dice":
        plt.ylim([-.02, 1.02])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.ylabel(display_name(metric))
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "4nm-vs-8nm_{0:}_{1:}.{2:}".format(metric, transparent, filetype)), transparent=transparent)
    plt.show()


def plot_all_vs_common_vs_single(metric: str,
                                 from_csv: bool = False,
                                 db: Optional[cosem_db.MongoCosemDB] = None,
                                 training_version: str = "v0003.2",
                                 tol_distance: int = 40,
                                 clip_distance: int = 200,
                                 filetype: str = "svg",
                                 transparent: bool = False,
                                 save: bool = False,
                                 to_csv: Optional[str] = None) -> None:
    """
    Plot the comparison of predicting on 4nm data with networks trained jointly on all labels, a large subset of labels
    or just a single or few labels.

    Args:
        metric: Metric to use for comparison.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.

    Returns:
        None.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons",
                                "all-vs-common-vs-single_test_{0:}.csv".format(metric))
        reader = csv.DictReader(open(csv_file, "r"))
        all_results = []
        for row in reader:
            all_results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        all_results = compare("all-vs-common-vs-single", db, [metric], tol_distance=tol_distance,
                              clip_distance=clip_distance, test=True)
    labels = []
    for result in all_results:
        result["label_all"] = short_names[result["label_all"]]
        result["label_common"] = short_names[result["label_common"]]
        result["label_single"] = short_names[result["label_single"]]
        labels.append(result["label_all"])
    labels = list(set(labels))
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for result in all_results:
    #     avgs[result["label_all"]].append(float(result["value_all"]))
    #     avgs[result["label_common"]].append(float(result["value_common"]))
    #     avgs[result["label_single"]].append(float(result["value_single"]))
    # for k, v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    # avgs_tuples = [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting(metric)]

    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)
    mids = list(range(1, 1 + len(labels)))
    x_values = []
    y_values = []
    colors = []
    off = 1 / 3.
    color_dict = {"all": "#1b9e77", "many": "#d95f02", "few": "#e6ab02"}
    crop_ds = {"111": "jrc_hela-3",
               "113": "jrc_hela-2",
               "110": "jrc_macrophage-2",
               "112": "jrc_jurkat-1"}
    csv_results = []
    for m, label in enumerate(labels):
        mid = mids[m]

        for result in all_results:
            if result["label_all"] == label:
                csv_entry = {"label": label, "dataset": crop_ds[result["crop_all"]]}
                x_values.append(mid - off)
                y_values.append(float(result["value_all"]))
                csv_entry["all"] = y_values[-1]
                colors.append(color_dict["all"])
                x_values.append(mid)
                y_values.append(float(result["value_common"]))
                csv_entry["many"] = y_values[-1]
                colors.append(color_dict["many"])
                x_values.append(mid + off)
                y_values.append(float(result["value_single"]))
                csv_entry["few"] = y_values[-1]
                colors.append(color_dict["few"])
                csv_results.append(csv_entry)

    vals = {}
    cols = {}
    for xx, yy, c in zip(x_values, y_values, colors):
        cols[xx] = c
        try:
            vals[xx].append(yy)
        except KeyError:
            vals[xx] = [yy]

    if to_csv is not None:
        csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                "figures", to_csv)
        fields = ["label", "dataset", "all", "many", "few"]
        csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(csv_results)

    plt.figure(figsize=(fig_width_per_label * len(labels), fig_height))
    ax = plt.gca()
    ax.set_axisbelow(True)
    for mid in mids[:-1]:
        ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
    plt.grid(axis="y", color="gray", linestyle=(0, (1, 5)), linewidth=.5)
    for xxk, yyv in vals.items():
        plt.plot([xxk] * len(yyv), yyv, color=cols[xxk], linewidth=2.5, alpha=.5)
    ax.scatter(x_values, y_values, c=colors, s=fs * ratio, marker=sym_test)
    for cat, col in color_dict.items():
        ax.scatter([], [], c=col, label=cat, s=fs * ratio, marker=sym_test)

    ax.set_xticks(mids)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(
        axis="both",
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False)
    ax.tick_params(axis="y", pad=15)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    if metric == "mean_false_distance":
        ax.set_yscale('log')
    plt.legend(frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=-.025)
    plt.xlim([0.5, max(mids) + 0.5])
    if metric == "dice":
        plt.ylim([-0.02, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel(display_name(metric))
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "all-vs-common-vs-single_{0:}_{1:}.{2:}".format(metric, transparent, filetype)),
            transparent=transparent)
    plt.show()


def plot_datasets(metric: str,
                  add_manual: bool = False,
                  from_csv: bool = False,
                  db: cosem_db.MongoCosemDB = None,
                  training_version: str = "v0003.2",
                  tol_distance: int = 40,
                  clip_distance: int = 200,
                  filetype: str = "svg",
                  transparent: bool = False,
                  save: bool = False,
                  to_csv: Optional[str] = None) -> None:
    """
    Plot for comparing results across the four different datasets/validation blocks. Plotting validation and test
    scores, optionally also manually optimized score.

    Args:
        metric: Metric to use for comparison ("dice" or "mean_false_distance")
        add_manual: Whether to include scores after manual evaluations in the plot.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.

    Returns:
        None.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file_test = os.path.join(csv_folder, "comparisons", "best-4nm_across-setups_test_{0:}.csv".format(metric))
        csv_file_validation = os.path.join(csv_folder, "comparisons",
                                           "best-4nm_across-setups_validation_{0:}.csv".format(metric))
        csv_file_manual = os.path.join(csv_folder, "comparisons",
                                       "metrics_across-setups_test_{0:}-vs-manual.csv".format(metric))
        reader_test = csv.DictReader(open(csv_file_test, "r"))
        reader_validation = csv.DictReader(open(csv_file_validation, "r"))
        reader_manual = csv.DictReader(open(csv_file_manual, "r"))

        test_results = []
        for row in reader_test:
            test_results.append(row)
        validation_results = []
        for row in reader_validation:
            validation_results.append(row)
        manual_results = []
        for row in reader_manual:
            manual_results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        test_results = compare("best-4nm", db, [metric], tol_distance=tol_distance,
                               clip_distance=clip_distance,
                               test=True, mode="across_setups")
        validation_results = compare("best-4nm", db, [metric], tol_distance=tol_distance,
                                     clip_distance=clip_distance, test=False, mode="across_setups")
        manual_results = compare("metrics", db, [metric, "manual"],
                                 tol_distance=tol_distance, clip_distance=clip_distance, mode="across_setups",
                                 test=True)
    labels = []
    crops = []
    for result in test_results:
        result["label"] = short_names[result["label"]]
        labels.append(result["label"])
        crops.append(result["crop"])
    for result in validation_results:
        result["label"] = short_names[result["label"]]
        labels.append(result["label"])
        crops.append(result["crop"])
    for result in manual_results:
        result["label_manual"] = short_names[result["label_manual"]]
    labels = list(set(labels))
    crops = ["113", "111", "112", "110"]
    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)

    x_values_test = []
    x_values_validation = []
    x_values_manual = []
    y_values_test = []
    y_values_validation = []
    y_values_manual = []
    colors = {"111": "#7570b3",  # lavender
              "113": "#66a61e",  # green
              "110": "#e7298a",  # pink
              "112": "#a6761d"  # ockery
              }
    crop_ds = {"111": "jrc_hela-3",
               "113": "jrc_hela-2",
               "110": "jrc_macrophage-2",
               "112": "jrc_jurkat-1"}

    patches = [
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["113"], label=crop_ds["113"]),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["111"], label=crop_ds["111"]),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["112"], label=crop_ds["112"]),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["110"], label=crop_ds["110"]),
    ]
    colors_test = []
    colors_validation = []
    colors_manual = []
    off = 0.24
    remove_label = []
    for label in labels:
        res_for_label = []
        for result in test_results:
            if result["label"] == label and result["value"] != "" and not np.isnan(float(result["value"])):
                res_for_label.append(result)
        for result in validation_results:
            if result["label"] == label and result["value"] != "" and not np.isnan(float(result["value"])):
                res_for_label.append(result)
        if len(res_for_label) < 8:
            remove_label.append(label)

    for label in remove_label:
        labels.pop(labels.index(label))
    csv_results = []
    mids = list(range(1, 1 + len(labels)))
    for m, label in enumerate(labels):
        mid = mids[m]
        for crop, idx in zip(crops, [-1.5, -0.5, 0.5, 1.5]):
            csv_entry = {"label": label, "dataset": crop_ds[crop]}
            for result in test_results:
                if result["label"] == label and result["crop"] == crop:
                    x_values_test.append(mid + idx * off)
                    if result["value"] == "":
                        y_values_test.append(np.nan)
                    else:
                        y_values_test.append(float(result["value"]))
                    csv_entry["test"] = y_values_test[-1]
                    colors_test.append(colors[crop])
            for result in validation_results:
                if result["label"] == label and result["crop"] == crop:
                    x_values_validation.append(mid + idx * off)
                    if result["value"] == "":
                        y_values_validation.append(np.nan)
                    else:
                        y_values_validation.append(float(result["value"]))
                    csv_entry["validation"] = y_values_validation[-1]
                    colors_validation.append(colors[crop])

            for result in manual_results:
                #print(label, result["label_manual"])
                if result["label_manual"] == label:
                    #print(label)
                    if result["crop_manual"] == crop:
                        #print(crop)
                        if result["raw_dataset_manual"] == "volumes/raw/s0":
                            #print(result["raw_dataset_manual"])
                            x_values_manual.append(mid + idx * off)
                            y_values_manual.append(float(result["value_manual"]))
                            colors_manual.append(colors[crop])
                            if add_manual:
                                csv_entry["manual"] = y_values_manual[-1]
            csv_results.append(csv_entry)


    vals = {}
    cols = {}
    for xx, yy, c in zip(x_values_test, y_values_test, colors_test):
        cols[xx] = c
        try:
            vals[xx].append(yy)
        except KeyError:
            vals[xx] = [yy]
    for xx, yy in zip(x_values_validation, y_values_validation):
        try:
            vals[xx].append(yy)
        except KeyError:
            vals[xx] = [yy]
    if add_manual:
        for xx, yy in zip(x_values_manual, y_values_manual):
            try:
                vals[xx].append(yy)
            except KeyError:
                vals[xx] = [yy]

    plt.figure(figsize=(fig_width_per_label * len(labels), fig_height))
    ax = plt.gca()
    if to_csv is not None:
        csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                "figures", to_csv)
        fields = ["label", "dataset", "test", "validation"]
        if add_manual:
            fields.append("manual")
        csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(csv_results)

    ax.set_axisbelow(True)
    for mid in mids[:-1]:
        ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
    plt.grid(axis='y', color="gray", linestyle=(0, (1, 5)), linewidth=.5)
    for xxk, yyv in vals.items():
        plt.plot([xxk] * len(yyv), yyv, color=cols[xxk], linewidth=2.5, alpha=.5)
    ax.scatter(x_values_test, y_values_test, c=colors_test, s=fs * ratio, marker=sym_test)
    ax.scatter(x_values_validation, y_values_validation, c=colors_validation, s=fs * ratio, marker=sym_val)

    ax.scatter([], [], c="k", label="test", s=fs * ratio, marker=sym_test)
    ax.scatter([], [], c="k", s=fs * ratio, label="validation", marker=sym_val)

    if add_manual:
        ax.scatter(x_values_manual, y_values_manual, c=colors_manual, marker=sym_manual, s=fs * ratio)
        ax.scatter([], [], c="k", label="manual", s=fs * ratio, marker=sym_manual)
    ax.set_xticks(mids)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which='both', bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis="y", pad=15)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    if metric == "mean_false_distance":
        ax.set_yscale('log')

    handles, handle_labels = ax.get_legend_handles_labels()
    plt.legend(handles=patches + handles, frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025)
    plt.xlim([0.5, max(mids) + 0.5])
    if metric == "dice":
        plt.ylim([-0.02, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel(display_name(metric))
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "datasets_{0:}_{1:}.{2:}".format(metric, transparent, filetype)), transparent=transparent)
    plt.show()


def plot_rawvsrefined_single(metric: str,
                             from_csv: bool = False,
                             db: cosem_db.MongoCosemDB = None,
                             training_version: str = "v0003.2",
                             tol_distance: int = 40,
                             clip_distance: int = 200,
                             filetype: str = "svg",
                             transparent: bool = False,
                             save: bool = False) -> None:
    """
    Plot the comparison of raw thresholded predictions and refined predictions for a single metric.

    Args:
        metric: Metric to use for comparison.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.

    Returns:
        None.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons", "raw-vs-refined_{metric:}.csv".format(metric=metric))
        reader = csv.DictReader(open(csv_file, "r"))
        results = []
        for row in reader:
            results.append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        results = compare("raw-vs-refined", db, [metric], tol_distance=tol_distance,
                          clip_distance=clip_distance)

    labels = []
    for result in results:
        result["label_raw"] = short_names[result["label_raw"]]
        result["label_refined"] = short_names[result["label_refined"]]
        result["value_raw"] = float(result["value_raw"])
        result["value_sort"] = float(result["value_sort"])
        result["value_refined"] = float(result["value_refined"])
        labels.append(result["label_raw"])

    labels = list(set(labels))
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for result in results:
    #     avgs[result["label_raw"]].append(result["value_sort"])
    # for k,v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    # avgs_tuples = [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting(metric)]

    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)
    mids = list(range(1, 1 + len(labels)))
    colors = {
        "111": "#7570b3",  # lavender
        "113": "#66a61e",  # green
        "110": "#e7298a",  # pink
        "112": "#a6761d"  # ockery
    }
    off = 0.25
    plt.figure(figsize=(fig_width_per_label * len(labels), fig_height))
    ax = plt.gca()
    ax.set_axisbelow(True)
    for mid in mids[:-1]:
        ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
    plt.grid(axis="y", color="gray", linestyle=(0, (1, 5)), linewidth=.5)

    ax.set_xticks(mids)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(
        axis="both",
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False)
    ax.tick_params(axis="y", pad=15)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")

    plt.xlim([0.5, max(mids) + 0.5])
    if metric == "dice" or metric == "recall" or metric == "precision":
        plt.ylim([-0.02, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if metric == "mean_false_distance":
        ax.set_yscale('log')
    for m, label in enumerate(labels):
        mid = mids[m]
        for result in results:
            if result["label_raw"] == label:
                if np.isnan(result["value_refined"]) and metric in ["dice", "precision", "recall"]:
                    refined = 0
                else:
                    refined = result["value_refined"]
                if np.isnan(result["value_raw"]) and metric in ["dice", "precision", "recall"]:
                    raw = 0
                else:
                    raw = result["value_raw"]
                ax.plot([mid - off, mid + off], [raw, refined], linewidth=2.5,
                        color=colors[
                            result["crop_raw"]], alpha=.5)
                ax.scatter([mid - off], [raw], color=colors[result["crop_raw"]], s=fs * ratio)
                ax.scatter([mid + off], [refined], facecolor="white", edgecolor=colors[result[
                    "crop_raw"]], s=fs * ratio)
    plt.ylabel(display_name(metric))
    patches = [
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["113"], label="jrc_hela-2"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["111"], label="jrc_hela-3"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["112"], label="jrc_jurkat-1"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["110"],
                      label="jrc_macrophage-2"),
    ]

    ax.scatter([], [], c="k", marker="o", label="raw", s=fs * ratio)
    ax.scatter([], [], facecolor="white", edgecolor="k", marker="o", label="refined", s=fs * ratio)
    handles, legend_labels = ax.get_legend_handles_labels()
    plt.legend(handles=patches + handles, frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025)

    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "raw-vs-refined_{0:}_{1:}.{2:}".format(metric, transparent, filetype)),
            transparent=transparent)
    plt.show()


def plot_raw_vs_refined(metrics: Sequence[str],
                        from_csv: bool = False,
                        db: cosem_db.MongoCosemDB = None,
                        training_version: str = "v0003.2",
                        tol_distance: int = 40,
                        clip_distance: int = 200,
                        filetype: str = "svg",
                        transparent: bool = False,
                        save: bool = False,
                        to_csv: Optional[str] = None) -> None:
    """
    Plot the comparison of raw thresholded predictions and refined predictions for a number of metrics.

    Args:
        metrics: Metrics to use for comparison.
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.

    Returns:
        None.
    """
    if len(metrics) == 1:
        plot_rawvsrefined_single(metrics[0], filetype=filetype, transparent=transparent, save=save)
        return
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.family'] = 'sans-serif'
    results = {}
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        csv_file = os.path.join(csv_folder, "comparisons", "raw-vs-refined_{metric:}.csv")
        for metric in metrics:
            results[metric] = []
            reader = csv.DictReader(open(csv_file.format(metric=metric), "r"))
            for row in reader:
                results[metric].append(row)
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        for metric in metrics:
            results[metric] = compare("raw-vs-refined", db, [metric], tol_distance=tol_distance,
                                      clip_distance=clip_distance)
    labels = []
    for n, metric in enumerate(metrics):
        for result in results[metric]:
            result["label_raw"] = short_names[result["label_raw"]]
            result["label_refined"] = short_names[result["label_refined"]]
            result["value_raw"] = float(result["value_raw"])
            result["value_refined"] = float(result["value_refined"])
            labels.append(result["label_raw"])
    labels = list(set(labels))
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for result in results[metrics[0]]:
    #     avgs[result["label_raw"]].append(result["value_raw"])
    # for k, v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    # avgs_tuples =  [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting(metrics[0])]

    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)
    mids = list(range(1, 1 + len(labels)))
    colors = {
        "111": "#7570b3",  # lavender
        "113": "#66a61e",  # green
        "110": "#e7298a",  # pink
        "112": "#a6761d"  # ockery
    }
    crop_ds = {"111": "jrc_hela-3",
               "113": "jrc_hela-2",
               "110": "jrc_macrophage-2",
               "112": "jrc_jurkat-1"}
    off = 0.25
    fig, axs = plt.subplots(len(metrics), sharex=True, sharey=True, figsize=(fig_width_per_label * len(labels),
                                                                             fig_height * len(metrics)))
    for ax in axs:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axs:
        for mid in mids[:-1]:
            ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
        ax.grid(axis="y", linestyle=(0, (1, 5)), color="gray", linewidth=.5)
        ax.set_xticks(mids)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax.tick_params(axis="y", pad=15)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
    plt.xlim([0.5, max(mids) + 0.5])
    if "dice" in metrics or "precision" in metrics or "recall" in metrics:
        plt.ylim([-0.04, 1.04])
    csv_results = []
    for ax_no, (ax, metric) in enumerate(zip(axs, metrics)):
        for m, label in enumerate(labels):
            mid = mids[m]
            for result in results[metric]:
                if result["label_raw"] == label:
                    csv_entry = {"label": label, "metric": display_name(metric), "dataset": crop_ds[result["crop_raw"]]}
                    if np.isnan(result["value_refined"]) and metric in ["dice", "precision", "recall"]:
                        refined = 0
                    else:
                        refined = result["value_refined"]
                    csv_entry["refined"] = refined

                    if np.isnan(result["value_raw"]) and metric in ["dice", "precision", "recall"]:
                        raw = 0
                    else:
                        raw = result["value_raw"]
                    csv_entry["raw"] = raw
                    ax.plot([mid - off, mid + off], [raw, refined], linewidth=2.5,
                            color=colors[result["crop_raw"]], alpha=.5)
                    ax.scatter([mid - off], [raw], color=colors[result["crop_raw"]], s=fs * ratio)
                    ax.scatter([mid + off], [refined], facecolor="white", edgecolor=colors[result[
                        "crop_raw"]], s=fs * ratio)
                    csv_results.append(csv_entry)
        if to_csv is not None:
            csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                    "figures", to_csv)
            fields = ["label", "metric", "dataset", "raw", "refined"]
            csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(csv_results)
        ax.set_ylabel(display_name(metric))
        if ax_no == 0:
            patches = [
                mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["113"],
                              label="jrc_hela-2"),
                mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["111"],
                              label="jrc_hela-3"),
                mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["112"],
                              label="jrc_jurkat-1"),
                mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["110"],
                              label="jrc_macrophage-2"),
            ]
            ax.scatter([], [], c="k", marker="o", label="raw", s=fs * ratio)
            ax.scatter([], [], facecolor="white", edgecolor="k", marker="o", label="refined", s=fs * ratio)
            handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles=patches + handles, frameon=False, prop={"size": fs}, labelspacing=.1,
                      handletextpad=.025)
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "raw-vs-refined_{0:}_{1:}.{2:}".format('_'.join(metrics), transparent, filetype)),
            transparent=transparent)
    plt.show()


def _assemble_metriccomparison_results(
        metric1: str,
        metric2: str,
        from_csv: bool = False,
        db: Optional[cosem_db.MongoCosemDB] = None,
        training_version: str = "v0003.2",
        tol_distance: int = 40,
        clip_distance: int = 200,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Union[str, float]]]]]]:
    results = dict()
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        across_setups_test = os.path.join(csv_folder, "comparisons",
                                          "metrics_across-setups_test_{0:}.csv".format("-vs-".join([metric1, metric2])))
        across_setups_validation = os.path.join(csv_folder, "comparisons",
                                                "metrics_across-setups_validation_{0:}.csv".format(
                                                    "-vs-".join([metric1, metric2])))
        per_setup_test = os.path.join(csv_folder, "comparisons",
                                      "metrics_per-setup_test_{0:}.csv".format("-vs-".join([metric1, metric2])))
        per_setup_validation = os.path.join(csv_folder, "comparisons",
                                            "metrics_per-setup_validation_{0:}.csv".format(
                                                "-vs-".join([metric1, metric2]))
                                            )

        reader_across_setups_test = csv.DictReader(open(across_setups_test, "r"))
        reader_across_setups_validation = csv.DictReader(open(across_setups_validation, "r"))
        reader_per_setup_test = csv.DictReader(open(per_setup_test, "r"))
        reader_per_setup_validation = csv.DictReader(open(per_setup_validation, "r"))
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        reader_across_setups_test = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                            clip_distance=clip_distance, mode="across_setups", test=True)
        reader_across_setups_validation = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                                  clip_distance=clip_distance, mode="across_setups", test=False)
        reader_per_setup_test = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                        clip_distance=clip_distance, mode="per_setup", test=True)
        reader_per_setup_validation = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                              clip_distance=clip_distance, mode="per_setup", test=False)

    for row in reader_across_setups_test:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s0":
            ff = open(os.path.join(csv_folder, "manual/compared_4nm_setups.csv"), "r")
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/subsampled/raw/0" or row[
            "raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            ff = open(os.path.join(csv_folder, "manual/compared_8nm_setups.csv"), "r")
        else:
            raise ValueError(
                "Unknown value {0:} in raw_dataset_{1:}".format(row["raw_dataset_{0:}".format(metric1)], metric1))
        for compare_row in csv.reader(ff):
            if compare_row[0] == row["label_{0:}".format(metric1)]:
                setups = compare_row[1:]
                break
        else:
            raise ValueError("missing entry for {0:} in {1:}".format(row["label_{0:}".format(metric1)], ff.name))
        if len(setups) == 1:
            continue
        row["label_{0:}".format(metric1)] = short_names[row["label_{0:}".format(metric1)]]
        row["label_{0:}".format(metric2)] = short_names[row["label_{0:}".format(metric2)]]
        if row["value_{0:}".format(metric1)] == "":
            row["value_{0:}".format(metric1)] = np.nan
        row["value_{0:}".format(metric1)] = float(row["value_{0:}".format(metric1)])
        if row["value_{0:}".format(metric2)] == "":
            row["value_{0:}".format(metric2)] = np.nan
        row["value_{0:}".format(metric2)] = float(row["value_{0:}".format(metric2)])
        if row["crop_{0:}".format(metric1)] not in results:
            results[row["crop_{0:}".format(metric1)]] = dict()
        if row["label_{0:}".format(metric1)] not in results[row["crop_{0:}".format(metric1)]]:
            results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]] = {"test": []}
        results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["test"].append(row)

    for crop, value_by_crop in results.items():
        for label, value_by_label in value_by_crop.items():
            print(len(value_by_label["test"]))
            assert len(value_by_label["test"]) == 1
            results[crop][label]["validation"] = [None, ] * len(value_by_label["test"])

    for row in reader_across_setups_validation:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        # if row["value_{0:}".format(metric1)] == "" or row["value_{0:}".format(metric2)] == "":
        #     continue
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s0":
            ff = open(os.path.join(csv_folder, "manual/compared_4nm_setups.csv"), "r")
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/subsampled/raw/0" or row["raw_dataset_{0:}".format(
                metric1)] == "volumes/raw/s1":
            ff = open(os.path.join(csv_folder, "manual/compared_8nm_setups.csv"), "r")
        else:
            raise ValueError(
                "Unknown value {0:} in raw_dataset_{1:}".format(row["raw_dataset_{0:}".format(metric1)], metric1))
        for compare_row in csv.reader(ff):
            if compare_row[0] == row["label_{0:}".format(metric1)]:
                setups = compare_row[1:]
                break
        else:
            raise ValueError("missing entry for {0:} in {1:}".format(row["label_{0:}".format(metric1)], ff.name))
        if len(setups) == 1:
            continue
        row["label_{0:}".format(metric1)] = short_names[row["label_{0:}".format(metric1)]]
        row["label_{0:}".format(metric2)] = short_names[row["label_{0:}".format(metric2)]]
        if row["value_{0:}".format(metric1)] == "":
            row["value_{0:}".format(metric1)] = np.nan
        row["value_{0:}".format(metric1)] = float(row["value_{0:}".format(metric1)])
        if row["value_{0:}".format(metric2)] == "":
            row["value_{0:}".format(metric2)] = np.nan
        row["value_{0:}".format(metric2)] = float(row["value_{0:}".format(metric2)])
        for k, res in enumerate(results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["test"]):
            if res["raw_dataset_{0:}".format(metric1)] == row["raw_dataset_{0:}".format(metric1)]:
                results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["validation"][k] = row
                break
    for row in reader_per_setup_test:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        row["label_{0:}".format(metric1)] = short_names[row["label_{0:}".format(metric1)]]
        row["label_{0:}".format(metric2)] = short_names[row["label_{0:}".format(metric2)]]
        if row["value_{0:}".format(metric1)] == "":
            row["value_{0:}".format(metric1)] = np.nan
        row["value_{0:}".format(metric1)] = float(row["value_{0:}".format(metric1)])
        if row["value_{0:}".format(metric2)] == "":
            row["value_{0:}".format(metric2)] = np.nan
        row["value_{0:}".format(metric2)] = float(row["value_{0:}".format(metric2)])
        if row["crop_{0:}".format(metric1)] not in results:
            results[row["crop_{0:}".format(metric1)]] = dict()
        if row["label_{0:}".format(metric1)] not in results[row["crop_{0:}".format(metric1)]]:
            results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]] = {"test": []}
        results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["test"].append(row)

    for crop, value_by_crop in results.items():
        for label, value_by_label in value_by_crop.items():
            tests = results[crop][label]["test"][1:]
            results[crop][label]["test"][1:] = sorted(tests, key=itemgetter("setup_{0:}".format(metric1),
                                                                            "raw_dataset_{0:}".format(metric1)))
            if "validation" in results[crop][label]:
                orig_len = len(results[crop][label]["validation"])
                new_len = len(results[crop][label]["test"])
                results[crop][label]["validation"].extend([None, ] * (new_len - orig_len))
            else:
                results[crop][label]["validation"] = [None, ] * len(value_by_label["test"])
    for row in reader_per_setup_validation:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        row["label_{0:}".format(metric1)] = short_names[row["label_{0:}".format(metric1)]]
        row["label_{0:}".format(metric2)] = short_names[row["label_{0:}".format(metric2)]]
        if row["value_{0:}".format(metric1)] == "":
            row["value_{0:}".format(metric1)] = np.nan
        row["value_{0:}".format(metric1)] = float(row["value_{0:}".format(metric1)])
        if row["value_{0:}".format(metric2)] == "":
            row["value_{0:}".format(metric2)] = np.nan
        row["value_{0:}".format(metric2)] = float(row["value_{0:}".format(metric2)])
        for k, res in enumerate(results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["test"]):
            if ((results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["validation"][k] is None)
                    and (res["raw_dataset_{0:}".format(metric1)] == row["raw_dataset_{0:}".format(metric1)])
                    and (res["setup_{0:}".format(metric1)] == row["setup_{0:}".format(metric1)])):
                results[row["crop_{0:}".format(metric1)]][row["label_{0:}".format(metric1)]]["validation"][k] = row
                break
    return results


def plot_metric_comparison_by_label(metric1: str,
                                    metric2: str,
                                    from_csv: bool = False,
                                    db: cosem_db.MongoCosemDB = None,
                                    training_version: str = "v0003.2",
                                    tol_distance: int = 40,
                                    clip_distance: int = 200,
                                    filetype: str = "svg",
                                    transparent: bool = False,
                                    save: bool = False,
                                    to_csv: Optional[str] = None) -> None:
    """
    Plot comparison of different metrics. For each label and dataset plot all available results. Results are reported
    via `metric1`, comparing optimizing setup/iteration via `metric1` and `metric2`.

    Args:
        metric1: Metric to use for comparison
        metric2: Metric to compare to, used as alternative for optimizing setup/iteration ( can include "manual")
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        save: whether to save the figure.
        to_csv: unless None, csv file with raw data shown in figure will be saved here.

    Returns:
        None.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    results = _assemble_metriccomparison_results(metric1, metric2, from_csv=from_csv, db=db,
                                                 training_version=training_version, tol_distance=tol_distance,
                                                 clip_distance=clip_distance)
    labels = []
    for crop, value_by_crop in results.items():
        labels.extend(list(value_by_crop.keys()))
    labels = list(set(labels))
    # results_with_manual = _assemble_metriccomparison_results("dice", "manual")
    # avgs = dict()
    # for label in labels:
    #     avgs[label] = []
    # for crop, value_by_crop in results_with_manual.items():
    #     for lbl, value_by_label in value_by_crop.items():
    #         for res in value_by_label["validation"]:
    #             avgs[lbl].append(res["value_{0:}".format("dice")])
    # for k,v in avgs.items():
    #     avgs[k] = np.nanmean(v)
    #
    # avgs_tuples = [(v, k) for k, v in avgs.items()]
    # labels = [l for _, l in sorted(avgs_tuples)]
    # labels = labels[::sorting("dice")]

    labels = sort_generic(labels, db=db, from_csv=from_csv, training_version=training_version,
                          tol_distance=tol_distance, clip_distance=clip_distance)

    max_datapoints = 0
    for crop, value_by_crop in results.items():
        for lbl, value_by_lbl in value_by_crop.items():
            if len(value_by_lbl["validation"]) > max_datapoints:
                max_datapoints = len(value_by_lbl["validation"])
    centering = max_datapoints % 2
    spacing = 1. / max_datapoints
    offsets = sorted(np.arange(spacing / 2, 1, spacing) - 0.5, key=lambda x: np.abs(x))
    mids = list(range(1, 1 + len(labels)))

    fig, axs = plt.subplots(4, sharex=True, sharey=True, figsize=(fig_width_per_label * len(labels), fig_height * 4))
    for ax in axs:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    crops_no_to_cell = {"113": "jrc_hela-2",
                        "111": "jrc_hela-3",
                        "112": "jrc_jurkat-1",
                        "110": "jrc_macrophage-2"}
    colors = {"111": "#7570b3",  # lavender
              "113": "#66a61e",  # green
              "110": "#e7298a",  # pink
              "112": "#a6761d"  # ockery
              }
    crops = ["113", "111", "112", "110"]
    for ax in axs:
        for mid in mids[:-1]:
            ax.axvline(mid + 0.5, linestyle=(0, (1, 5)), color="gray", linewidth=.5)
        ax.grid(axis="y", color="gray", linestyle=(0, (1, 5)), linewidth=.5)
        ax.set_xticks(mids)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax.tick_params(axis="y", pad=15)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs, rotation_mode="anchor")
        if metric1 == "mean_false_distance":
            ax.set_yscale("log")
    plt.xlim([0.5, max(mids) + 0.5])
    if metric1 == "dice":
        plt.ylim([-0.04, 1.04])
        legend_location = "upper right"
    elif metric1 == "mean_false_distance":
        legend_location = "lower right"
    if metric2 == "dice":
        metric_label = "F1 Score"
    elif metric2 == "mean_false_distance":
        metric_label = "Mean False Distance"
    elif metric2 == "manual":
        metric_label = "manual"
    csv_results = []
    for ax_no, (ax, crop) in enumerate(zip(axs, crops)):
        for m, label in enumerate(labels):

            try:
                vals = results[crop][label]["validation"]
                tests = results[crop][label]["test"]
            except KeyError:
                continue
            assert len(vals) == len(tests)
            # print(crop, label)
            # print("VALS", vals)
            # print("TESTS", tests)
            if len(vals) % 2 != centering:
                centering_shift = spacing / 2
            else:
                centering_shift = 0
            for val, test, off in zip(vals, tests, offsets):
                if val is None:
                    continue
                csv_entry = {"dataset": crops_no_to_cell[crop], "label": label}
                metric1_value_validation = val["value_{0:}".format(metric1)]
                csv_entry["validation"] = metric1_value_validation
                metric1_value_test = test["value_{0:}".format(metric1)]
                metric2_value = val["value_{0:}".format(metric2)]
                csv_entry[metric_label] = metric2_value
                ax.scatter(mids[m] + off + centering_shift, metric1_value_validation, c=colors[crop], s=fs * ratio,
                           marker=sym_val)
                ax.scatter(mids[m] + off + centering_shift, metric2_value, c=colors[crop], s=fs * ratio,
                           marker=sym_manual)
                # ax.scatter(mids[m]+off+centering_shift, metric1_value_test, c=colors[crop], s=fs*ratio,
                # marker=sym_test)
                ax.plot([mids[m] + off + centering_shift] * 2, [metric1_value_validation, metric2_value],  # ,
                        # metric1_value_test],
                        color=colors[crop], linewidth=2.5, alpha=.5)
                csv_results.append(csv_entry)
        patch = [mlines.Line2D([], [], marker="s", markersize=fs * 0.5, linewidth=0, color=colors[crop],
                               label=crops_no_to_cell[crop]), ]
        if to_csv is not None:
            csv_file = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version,
                                    "figures", to_csv)
            fields = ["dataset", "label", "validation", metric_label]
            csv_writer = csv.DictWriter(open(csv_file, "w"), fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(csv_results)
        if ax_no == 0:
            # ax.scatter([], [], c="k", label="test", s=fs * ratio, marker=sym_test)
            ax.scatter([], [], c="k", s=fs * ratio, label="validation", marker=sym_val)

            ax.scatter([], [], c="k", marker=sym_manual, label=metric_label, s=fs * ratio)
            handles, handle_labels = ax.get_legend_handles_labels()

            # fake entries for legend to make split between columns sensible
            pseudo_patches = [mlines.Line2D([], [], markersize=0, linewidth=0, color=colors[crop], label=""), ]
            if "upper" in legend_location:
                assembled_handles = handles + patch + pseudo_patches
            elif "lower" in legend_location:
                assembled_handles = handles + pseudo_patches + patch
            ax.legend(handles=assembled_handles, frameon=False, prop={"size": fs}, labelspacing=.1,
                      handletextpad=.025, ncol=2, loc=legend_location)
        else:
            ax.legend(handles=patch, frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025,
                      loc=legend_location)

    # add fake plot to get joint ylabel
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    ax.tick_params(axis="y", pad=15)
    ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

    plt.ylabel(display_name(metric1))
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "metrics-by-label_{0:}-vs-{1:}_{2:}.{3:}".format(metric1, metric2, transparent, filetype)),
            transparent=transparent
        )
    plt.show()


def plot_metric_comparison(metric1: str,
                           metric2: str,
                           from_csv: bool = False,
                           db: cosem_db.MongoCosemDB = None,
                           training_version: str = "v0003.2",
                           tol_distance: int = 40,
                           clip_distance: int = 200,
                           filetype: str = "svg",
                           transparent: bool = False,
                           label: Optional[str] = None,
                           save: bool = False) -> None:
    """
    Plot comparison of different metrics. Plot all available validation results, color coded by dataset. Plot
    validation scores (`metric1`) against the difference in score generated by using a different metric (`metric2`) for
    optimizing the configuration (setup/iteration).

    Args:
        metric1: Metric to use for comparison.
        metric2: Metric to compare to, used as alternative for optimizing setup/iteration (can include "manual").
        from_csv: Read results from csv file in standardized location.
        db: Database with evaluation results, needed if `from_csv` is `False`.
        training_version: Version of trainings to compare.
        tol_distance: tolerance distance when using metric with tolerance distance, otherwise not used.
        clip_distance: clip distance when using metric with clip distance, otherwise not used.
        filetype: Filetype for saving the figure.
        transparent: Whether to save the figure with a transparent background.
        label: Only plot results for this label. If None, plot all.
        save: whether to save the figure.

    Returns:
        None.
    """
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    if from_csv:
        csv_folder = os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version)
        across_setups_test = os.path.join(csv_folder, "comparisons",
                                          "metrics_across-setups_test_{0:}.csv".format("-vs-".join([metric1, metric2])))
        across_setups_validation = os.path.join(csv_folder, "comparisons",
                                                "metrics_across-setups_validation_{0:}.csv".format(
                                                    "-vs-".join([metric1, metric2])))
        per_setup_test = os.path.join(csv_folder, "comparisons",
                                      "metrics_per-setup_test_{0:}.csv".format("-vs-".join([metric1, metric2])))
        per_setup_validation = os.path.join(csv_folder, "comparisons",
                                            "metrics_per-setup_validation_{0:}.csv".format(
                                                "-vs-".join([metric1, metric2])
                                            ))

        reader_across_setups_test = csv.DictReader(open(across_setups_test, "r"))
        reader_across_setups_validation = csv.DictReader(open(across_setups_validation, "r"))
        reader_per_setup_test = csv.DictReader(open(per_setup_test, "r"))
        reader_per_setup_validation = csv.DictReader(open(per_setup_validation, "r"))
    else:
        assert db is not None, "Need access to database if results aren't read from csv file."
        reader_across_setups_test = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                            clip_distance=clip_distance, mode="across_setups", test=True)
        reader_across_setups_validation = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                                  clip_distance=clip_distance, mode="across_setups", test=False)
        reader_per_setup_test = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                        clip_distance=clip_distance, mode="per_setup", test=True)
        reader_per_setup_validation = compare("metrics", db, [metric1, metric2], tol_distance=tol_distance,
                                              clip_distance=clip_distance, mode="per_setup", test=False)
    plt.figure(figsize=(fig_width_per_label * 2, fig_width_per_label * 2))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    if sorting(metric1) == 1:
        ax.invert_yaxis()
    plt.grid(axis="both", color="gray", linestyle=(0, (1, 5)), linewidth=.5)
    colors = {"110": "#7570b3", "111": "#66a61e", "112": "#e7298a", "113": "#a6761d"}

    test_results = []
    for row in reader_across_setups_test:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s0":
            ff = open(os.path.join(csv_folder, "manual/compared_4nm_setups.csv"), "r")
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/subsampled/raw/0":
            ff = open(os.path.join(csv_folder, "manual/compared_8nm_setups.csv"), "r")
        else:
            raise ValueError("Unknown value {0:} in raw_dataset_{1:}".format(row["raw_dataset_{0:}".format(metric1)],
                                                                             metric1))
        for compare_row in csv.reader(ff):

            if compare_row[0] == row["label_{0:}".format(metric1)]:
                setups = compare_row[1:]
                break
        else:
            raise ValueError("missing entry for {0:} in {1:}".format(row["label_{0:}".format(metric1)], ff.name))
        if len(setups) == 1:
            continue
        test_results.append(row)
    for row in reader_per_setup_test:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue

    x_values_test = []
    y_values_test = []
    cols_test = []
    for res in test_results:
        if res["value_{0:}".format(metric1)] == "" or res["value_{0:}".format(metric2)] == "":
            continue
        if label is not None and res["label_{0:}".format(metric1)] != label:
            continue
        x_values_test.append(float(res["value_{0:}".format(metric1)]))
        y_values_test.append(float(res["value_{0:}".format(metric1)]) - float(res["value_{0:}".format(metric2)]))
        cols_test.append(colors[res["crop_{0:}".format(metric1)]])
        if abs(y_values_test[-1]) > 0.3:
            print(res["label_{0:}".format(metric1)])

    validation_results = []
    for row in reader_across_setups_validation:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s0":
            ff = open(os.path.join(csv_folder, "manual/compared_4nm_setups.csv"), "r")
        elif row["raw_dataset_{0:}".format(metric1)] == "volumes/subsampled/raw/0":
            ff = open(os.path.join(csv_folder, "manual/compared_8nm_setups.csv"), "r")
        else:
            raise ValueError("Unknown value {0:} in raw_dataset_{1:}".format(row["raw_dataset_{0:}".format(metric1)],
                                                                             metric1))
        for compare_row in csv.reader(ff):
            if compare_row[0] == row["label_{0:}".format(metric1)]:
                setups = compare_row[1:]
                break
        else:
            raise ValueError("missing entry for {0:} in {1:}".format(row["label_{0:}".format(metric1)], ff.name))
        if len(setups) == 1:
            continue
        validation_results.append(row)
    for row in reader_per_setup_validation:
        if row["raw_dataset_{0:}".format(metric1)] == "volumes/raw/s1":
            continue
        validation_results.append(row)

    x_values_validation = []
    y_values_validation = []
    cols_validation = []
    for res in validation_results:
        if res["value_{0:}".format(metric1)] == "" or res["value_{0:}".format(metric2)] == "":
            continue
        if label is not None and res["label_{0:}".format(metric1)] != label:
            continue
        x_values_validation.append(float(res["value_{0:}".format(metric1)]))
        y_values_validation.append(float(res["value_{0:}".format(metric1)]) - float(res["value_{0:}".format(metric2)]))
        cols_validation.append(colors[res["crop_{0:}".format(metric1)]])
        if y_values_validation[-1] > 0.3:
            print(res["label_{0:}".format(metric1)])

    ax.scatter(x_values_validation, y_values_validation, c=colors, s=fs * ratio, marker=sym_val)

    patches = [
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["113"], label="jrc_hela-2"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["111"], label="jrc_hela-3"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["112"], label="jrc_jurkat-1"),
        mlines.Line2D([], [], marker="s", markersize=fs * 0.6, linewidth=0, color=colors["110"],
                      label="jrc_macrophage-2"),
    ]
    test_label = "test/manual" if metric2 == "manual" else "test/test"
    validation_label = "validation/manual" if metric2 == "manual" else "validation/validation"
    ax.scatter([], [], c="k", marker=sym_test, label=test_label, s=fs * ratio)
    ax.scatter([], [], c="k", marker=sym_manual, s=fs * ratio, label=validation_label)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=patches + handles, frameon=False, prop={"size": fs}, labelspacing=.1, handletextpad=.025)

    plt.ylabel(r"$\Delta$ {0:}".format(display_name(metric1)))
    plt.xlabel(display_name(metric1))

    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
    ax.tick_params(axis="y", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if metric1 == "dice":
        plt.ylim([-.85, .85])
    plt.tight_layout()
    if save:
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.15)
        plt.savefig(
            os.path.join(config_loader.get_config()["organelles"]["evaluation_path"], training_version, "figures",
                         "datasets_{0:}-vs-{1:}_{2:}.{3:}".format(metric1, metric2, transparent, filetype)),
            transparent=transparent)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser("Make paper comparison plots")
    parser.add_argument("plot", type=str, choices=["s1-vs-sub", "best_per_label", "4nm-vs-8nm",
                                                   "all-vs-common-vs-single", "generalization", "datasets", "metrics",
                                                   "metrics-by-label", "raw-vs-refined"])
    # can be run for any of the metrics, but plot layout is generally optimized for dice and/or mean_false_distance,
    # other metrics prob need some adaptations
    parser.add_argument("--metric", type=str, default=None, help="metric to evaluate",
                        choices=list(em.value for em in EvaluationMetrics) + ["manual"], nargs="+")
    parser.add_argument("--label", type=str, default=None, help="restrict metric comparison plot to one label")
    parser.add_argument("--manual", action="store_true", help="add manual results to datasets plot")
    parser.add_argument("--from_csv", action="store_true", help="read results from csv files in standardized location")
    parser.add_argument("--tol_distance", type=int, default=40,
                        help="Parameter used for tolerated false distances for relevant metrics")
    parser.add_argument("--clip_distance", type=int, default=200,
                        help="Parameter used for clipped false distances for relevant metrics")
    parser.add_argument("--training_version", type=str, default="v0003.2", help="Version of training")
    parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth")
    parser.add_argument("--filetype", type=str, default="svg", help="format to save plot in")
    parser.add_argument("--transparent", action="store_true", help="whether to save with transparent background")
    parser.add_argument("--save", action="store_true", help="whether to save plot to file")
    parser.add_argument("--to_csv", type=str, default=None, help="Csv file to save raw data values to")

    args = parser.parse_args()
    print(args)
    if not args.from_csv:
        db = cosem_db.MongoCosemDB(training_version=args.training_version, gt_version=args.gt_version)
    else:
        db = None
    if args.plot == "metrics":
        assert len(args.metric) == 2
        plot_metric_comparison(args.metric[0], args.metric[1], from_csv=args.from_csv, db=db,
                               training_version=args.training_version, tol_distance=args.tol_distance,
                               clip_distance=args.clip_distance, filetype=args.filetype, transparent=args.transparent,
                               label=args.label, save=args.save)
    elif args.plot == "metrics-by-label":
        assert len(args.metric) == 2
        plot_metric_comparison_by_label(args.metric[0], args.metric[1], from_csv=args.from_csv, db=db,
                                        training_version=args.training_version, tol_distance=args.tol_distance,
                                        clip_distance=args.clip_distance, filetype=args.filetype,
                                        transparent=args.transparent, save=args.save, to_csv=args.to_csv)
    elif args.plot == "raw-vs-refined":
        pass
    else:
        args.metric = args.metric[0]
    if args.plot == "s1-vs-sub":
        plot_s1_vs_sub(args.metric, from_csv=args.from_csv, db=db, training_version=args.training_version,
                       tol_distance=args.tol_distance, clip_distance=args.clip_distance, filetype=args.filetype,
                       transparent=args.transparent,
                       save=args.save)
    elif args.plot == "4nm-vs-8nm":
        plot_4nm_vs_8nm(args.metric, from_csv=args.from_csv, db=db, training_version=args.training_version,
                        tol_distance=args.tol_distance, clip_distance=args.clip_distance, filetype=args.filetype,
                        transparent=args.transparent, save=args.save, to_csv=args.to_csv)
    elif args.plot == "all-vs-common-vs-single":
        plot_all_vs_common_vs_single(args.metric, from_csv=args.from_csv, db=db, training_version=args.training_version,
                                     tol_distance=args.tol_distance, clip_distance=args.clip_distance,
                                     filetype=args.filetype, transparent=args.transparent, save=args.save,
                                     to_csv=args.to_csv)
    elif args.plot == "generalization":
        plot_generalization(args.metric, args.label, from_csv=args.from_csv, db=db,
                            training_version=args.training_version, tol_distance=args.tol_distance,
                            clip_distance=args.clip_distance, filetype=args.filetype, transparent=args.transparent,
                            save=args.save, to_csv=args.to_csv)
    elif args.plot == "datasets":
        plot_datasets(args.metric, args.manual, from_csv=args.from_csv, db=db, training_version=args.training_version,
                      tol_distance=args.tol_distance, clip_distance=args.clip_distance, filetype=args.filetype,
                      transparent=args.transparent, save=args.save, to_csv=args.to_csv)
    elif args.plot == "raw-vs-refined":
        plot_raw_vs_refined(args.metric, from_csv=args.from_csv, db=db, training_version=args.training_version,
                            tol_distance=args.tol_distance, clip_distance=args.clip_distance, filetype=args.filetype,
                            transparent=args.transparent, save=args.save, to_csv=args.to_csv)


if __name__ == "__main__":
    main()
