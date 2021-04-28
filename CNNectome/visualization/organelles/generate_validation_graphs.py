from CNNectome.utils import hierarchy
from CNNectome.utils.crop_utils import check_label_in_crop
from CNNectome.utils.cosem_db import MongoCosemDB
from CNNectome.utils.setup_utils import get_unet_setup, detect_8nm
from CNNectome.validation.organelles.check_consistency import convergence_iteration, max_iteration_for_analysis
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import os
import sys
from typing import Optional


def plot_val(db: MongoCosemDB,
             setup: str,
             labelname: str,
             file: Optional[str],
             threshold: int = 127) -> None:
    """
    Plot validation graph for a specific setup and label. Can be saved by specifying a `file`.

    Args:
        db: Database with crop information and evaluation results.
        setup: Setup to plot validation results for.
        labelname: Label to plot validation results for.
        file: File to save validation plot to, if None show plot instead.
        threshold: Threshold to be applied on top of raw predictions to generate binary segmentations for evaluation.

    Returns:
        None.
    """
    print(setup, labelname)
    valcrops = db.get_all_validation_crops()

    if len(valcrops) != 4:
        raise NotImplementedError("Number of validation crops has changed so plotting layout has to be updated")

    if detect_8nm(setup):
        raw_datasets = ["volumes/subsampled/raw/0", "volumes/raw/s1"]
    else:
        raw_datasets = ["volumes/raw"]

    col = db.access("evaluation", db.training_version)
    # query all relevant results
    query = {"setup": setup,
             "label": labelname,
             "refined": False,
             "iteration": {"$mod": [25000, 0]},
             "threshold": threshold}
    results = dict()
    max_its = dict()
    for crop in valcrops:
        query["crop"] = crop["number"]
        if col.find_one(query) is None:
            continue
        results[crop["number"]] = dict()
        max_its[crop["number"]] = dict()
        for raw_ds in raw_datasets:
            query["raw_dataset"] = raw_ds
            results[crop["number"]][raw_ds] = dict()
            max_its[crop["number"]][raw_ds] = dict()

            max_it_actual = convergence_iteration(query, db)
            max_it_min700 = max_iteration_for_analysis(query, db, conv_it=max_it_actual)

            max_its[crop["number"]][raw_ds]["actual"] = max_it_actual
            max_its[crop["number"]][raw_ds]["min700"] = max_it_min700
            for metric in ["dice", "mean_false_distance"]:
                query["metric"] = metric

                col = db.access("evaluation", db.training_version)

                scores = list(col.aggregate([
                    {"$match": query},
                    {"$sort": {"iteration": 1}},
                    {"$project": {"iteration": 1, "_id": 0, "value": 1}}
                ]))
                results[crop["number"]][raw_ds][metric] = scores

    colors = {"dice": "tab:green", "mean_false_distance": "tab:blue"}
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(30, 15))
    if len(raw_datasets) > 1:
        plt.plot([], [], marker='.', ms=1.2, linestyle="-", color="k", label="subsampled")
        plt.plot([], [], marker='.', ms=1.2, linestyle="--", color="k", label="averaged")
    plt.plot([], [], linestyle="-", color="tab:red", label="max iteration (min 700k)")
    plt.plot([], [], linestyle="-", color="tab:pink", label="max iteration (no min)")
    fig.legend(loc='upper right', frameon=False, prop={"size": 18})

    plt.suptitle("{setup:}  -  {label:}".format(setup=setup, label=labelname), fontsize=22)

    for crop, ax in zip(valcrops, axs.flatten()):
        try:
            crop_res = results[crop["number"]]
        except KeyError:
            continue

        ax2 = ax.twinx()
        for raw_ds, ls in zip(raw_datasets, ["-", "--"]):
            x_vs_dice = [r["iteration"] for r in crop_res[raw_ds]["dice"]]
            y_vs_dice = [1 - r["value"] for r in crop_res[raw_ds]["dice"]]
            x_vs_mfd = [r["iteration"] for r in crop_res[raw_ds]["mean_false_distance"]]
            y_vs_mfd = [r["value"] for r in crop_res[raw_ds]["mean_false_distance"]]

            ax.plot(x_vs_mfd, y_vs_mfd, linestyle=ls, color=colors["mean_false_distance"], marker='o', ms=3)
            ax2.plot(x_vs_dice, y_vs_dice, linestyle=ls, color=colors["dice"], marker='o', ms=3)
            if max_its[crop["number"]][raw_ds]["min700"][1]:
                ax.axvline(max_its[crop["number"]][raw_ds]["min700"][0], linestyle=ls, color="tab:red")
                if max_its[crop["number"]][raw_ds]["min700"][0] != max_its[crop["number"]][raw_ds]["actual"][0]:
                    ax.axvline(max_its[crop["number"]][raw_ds]["actual"][0], linestyle=ls, color="tab:pink")

            ax.set_xlabel("iteration", fontsize=18)
            ax.set_title(crop["number"], fontsize=18)
            ax.xaxis.set_major_formatter(ticker.EngFormatter())

            ax.set_ylabel("MFD", color=colors["mean_false_distance"], fontsize=18)
            ax.tick_params(axis="y", labelcolor=colors["mean_false_distance"])
            ax.set_ylim(bottom=0)

            ax2.set_ylabel("1 - dice", color=colors["dice"], fontsize=18)
            ax2.tick_params(axis="y", labelcolor=colors["dice"])
            ax2.set_ylim([0, 1])

            ax.tick_params(axis="both", which="major", labelsize=18)
            ax2.tick_params(axis="both", which="major", labelsize=18)

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()


def plot_val_all_setups(db: MongoCosemDB,
                        path: str,
                        threshold: int = 127,
                        filetype: str = "pdf") -> None:
    """
    Plot validation graphs for all the setups and labels. Will be saved to files in `path`.

    Args:
        db: Database with crop information and evaluation results.
        path: Path in which to save all the plots.
        threshold: Threshold to be applied on top of raw predictions to generate binary segmentations for evaluation.
        filetype: Filetype for saving plots.

    Returns:
        None.
    """
    setups = ["setup01", "setup03", "setup04", "setup25", "setup26.1", "setup27.1", "setup28", "setup31", "setup32",
              "setup33", "setup34", "setup35", "setup36", "setup37", "setup38", "setup39", "setup40", "setup43",
              "setup44", "setup45", "setup46", "setup47", "setup48", "setup49.1", "setup50", "setup56", "setup59",
              "setup61", "setup62", "setup63", "setup64"]

    for setup in setups:
        plot_val_all_labels(db, setup, path, threshold, filetype)


def plot_val_all_labels(db: MongoCosemDB,
                        setup: str,
                        path: str,
                        threshold: int = 127,
                        filetype: str = "pdf"):
    """
    Plot validation graphs for all labels corresponding to a specific setup. Will be saved to files in `path`.

    Args:
        db: Database with crop information and evaluation results.
        setup: Setup to plot validation results for.
        path: Path in which to save all the plots.
        threshold: Threshold to be applied on top of raw predictions to generate binary segmentations for evaluation.
        filetype: Filetype for saving plots.
    """
    valcrops = db.get_all_validation_crops()
    labels = get_unet_setup(setup).labels
    for lbl in labels:
        in_crop = [check_label_in_crop(lbl, crop) for crop in valcrops]
        if any(in_crop):
            file = os.path.join(path, "{label:}_{setup:}.{filetype:}".format(label=lbl.labelname,
                                                                             setup=setup, filetype=filetype))
            plot_val(db, setup, lbl.labelname, file, threshold=threshold)


def main() -> None:
    main_parser = argparse.ArgumentParser("Plot validation graphs")
    parser = main_parser.add_subparsers(dest="script", help="")

    all_parser = parser.add_parser("all_setups", help="Validation graphs for all default setups.")

    all_parser.add_argument("--threshold", type=int, default=127,
                            help=("Threshold to be applied on top of raw predictions to generate binary "
                                  "segmentations for evaluation."))
    all_parser.add_argument("--path", type=str, default='.', help="Path to save validation graphs to.")
    all_parser.add_argument("--filetype", type=str, default='pdf', help="Filetype for validation plots.")
    all_parser.add_argument("--training_version", type=str, default="v0003.2",
                            help="Version of training for which to plot validation graphs.")
    all_parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth to consider.")

    setup_parser = parser.add_parser("all_labels", help="Validation graphs for all labels of a specific setup.")
    setup_parser.add_argument("setup", type=str, help="Setup to make validation graphs for.")
    setup_parser.add_argument("--path", type=str, default="Path to save validation graphs to.")
    setup_parser.add_argument("--filetype", type=str, default="pdf", help="Filetype for validation plots.")
    setup_parser.add_argument("--threshold", type=int, default=127,
                              help=("Threshold to be applied on top of raw predictions to generate binary "
                                    "segmentations for evaluation."))
    setup_parser.add_argument("--training_version", type=str, default="v0003.2",
                              help="Version of training for which to plot validation graphs.")
    setup_parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth to consider.")

    single_parser = parser.add_parser("single", help="Validation graph for a specific setup and label.")
    single_parser.add_argument("setup", type=str, help="Setup to make validation graph for.")
    single_parser.add_argument("label", type=str, help="Label to make validation graph for.")
    single_parser.add_argument("--file", type=str, help="File to save validation graph to. (Full path)")
    single_parser.add_argument("--threshold", type=int, default=127,
                               help=("Threshold to be applied on top of raw predictions to generate binary "
                                     "segmentations for evaluation."))
    single_parser.add_argument("--training_version", type=str, default="v0003.2",
                               help="Version of training for which to plot validation graphs.")
    single_parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth to consider.")

    args = main_parser.parse_args()
    db = MongoCosemDB(training_version=args.training_version, gt_version=args.gt_version)
    if args.script == "all_setups":
        plot_val_all_setups(db, args.path, threshold=args.threshold, filetype=args.filetype)
    elif args.script == "all_labels":
        plot_val_all_labels(db, args.setup, args.path, threshold=args.threshold, filetype=args.filetype)
    else:
        plot_val(db, args.setup, args.label, args.file, threshold=args.threshold)


if __name__ == "__main__":
    main()
