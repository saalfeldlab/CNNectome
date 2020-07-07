from CNNectome.validation.organelles.manual_evals import *
from CNNectome.validation.organelles.segmentation_metrics import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

colors = ["green", "magenta", "orange"]

def plot_hist(db_username, db_password, metric, domain, cropno):
    results = get_differences(db_username, db_password, cropno, metric, domain=domain)
    differences = [abs(r["manual_best"] - r["auto_best"]) for r in results]
    sns.set_style('darkgrid')
    plt.hist(differences, histtype="stepfilled", alpha=.7,bins=20)
    plt.xlabel(r'$\Delta$ ' + metric)
    plt.xlim(left=0)
    plt.show()


def plot_scatter(db_username, db_password, metric, domain, cropno):
    results = []
    for d in domain:
        results.extend(get_differences(db_username, db_password, cropno, metric, domain=d))
    sns.set_style('darkgrid')
    fs = 40

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

    autos = [r["auto_best"] for r in results]
    manuals = [r["manual_best"] for r in results]
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_aspect("equal")
    print([(a,m) for a,m in zip(autos, manuals)])
    plt.scatter(autos, manuals,s=180)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'w-', zorder=0, linewidth=3)
    plt.xlim(lims)
    plt.ylim(lims)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    plt.ylabel(metric + " for best according to manual evaluation", fontsize=fs)
    plt.xlabel(metric + " for best according to automatic evaluation", fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/figures/COSEM/manual_vs_auto_"+metric+".png"), transparent=True)
    plt.show()


def plot_stem(db_username, db_password, compare_metric, eval_metric, domain, cropno, cropnames, top, bottom, left, right):
    results = {}
    for cno in cropno:
        results[cno] = []
    metrics = [eval_metric, ]
    if compare_metric != "manual":
        metrics.append(compare_metric)
    for cno in cropno:
        for d in domain:
            results[cno].extend(get_differences(db_username, db_password, cno, metrics, domain=d))
    sns.set_style('darkgrid')
    fs = 20
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
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.tick_params(axis="y", which="major", labelsize=fs)
    ax.tick_params(axis="x", which="major", labelsize=fs)

    for cno, col, name in zip(cropno, colors, cropnames):
        same = [r["{eval_metric:}_by_{eval_metric:}".format(eval_metric=eval_metric)] for r in results[cno]]
        compare = [r["{eval_metric:}_by_{compare_metric:}".format(eval_metric=eval_metric, compare_metric=compare_metric)] for r in results[cno]]

        diffs = [c-s for s,c in zip(same, compare)]
        print([(d, r["setup"], r["labelname"], r["s1"]) for d, r in zip(diffs, results[cno])])
        if np.sum(diffs) < 0 and not ax.yaxis_inverted():
            ax.invert_yaxis()

        markers, stems, base = plt.stem(same, diffs, use_line_collection=True, markerfmt='o', basefmt=" ", label=name)
        plt.setp(stems, 'linewidth', 3)
        plt.setp(markers, 'markersize', 10)
        plt.setp(markers, 'color', col)
        plt.setp(markers, 'markeredgewidth', 2)
        plt.setp(stems, 'color', col)

    l, u = limits(eval_metric)
    if l is not None:
        ax.set_xlim(left=l)
    if u is not None:
        ax.set_xlim(right=u)
    if left is not None:
        ax.set_xlim(left=left)
    if right is not None:
        ax.set_xlim(right=right)
    b, t = ax.get_ylim()
    print(b, t)
    if top is not None:
        ax.set_ylim(top=top)
    if bottom is not None:
        ax.set_ylim(bottom=bottom)
    plt.xlabel(eval_metric + " for best according to {0:}".format(eval_metric), fontsize=fs)

    if compare_metric == "manual":
        plt.ylabel(r"$\Delta$ {0:} for best according to manual evaluation".format(eval_metric), fontsize=fs)
    else:
        plt.ylabel(r"$\Delta$ {0:} for best according to {1:}".format(eval_metric, compare_metric), fontsize=fs)

    leg = ax.legend(fontsize=fs, prop={"size": 30})
    for text in leg.get_texts():
        plt.setp(text, color="w")

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/figures/COSEM/{0:}_vs_{1:}_stem.png".format(eval_metric, compare_metric)),
                transparent=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_username", type=str, help="username for the database")
    parser.add_argument("--db_password", type=str, help="password for the database")

    parser.add_argument("--compare_metric", type=str, default=None, help="metric to compare to for picking best",
                        choices=list(em.value for em in EvaluationMetrics) + ["manual"])
    parser.add_argument("--eval_metric", type=str, default=None, help="metric to evaluate",
                        choices=list(em.value for em in EvaluationMetrics))

    parser.add_argument("--domain", type=str, choices=["setup", "iteration"], default="iteration", nargs="+")
    parser.add_argument("--cropno", type=int, default=111, nargs="+")
    parser.add_argument("--top", type=float, default=None)
    parser.add_argument("--bottom", type=float, default=None)
    parser.add_argument("--left", type=float, default=None)
    parser.add_argument("--right", type=float, default=None)
    parser.add_argument("--cropnames", type=str, nargs="+")
    args = parser.parse_args()
    plot_stem(args.db_username, args.db_password, args.compare_metric, args.eval_metric, args.domain, args.cropno, args.cropnames, args.top, args.bottom, args.left, args.right)


if __name__ == "__main__":
    main()