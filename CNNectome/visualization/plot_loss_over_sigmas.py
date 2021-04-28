import argparse
import json
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import os
import re


def find_sigmas(directory):
    sigmas = []
    p = re.compile("costs_sigma(([0-9]*\.[0-9]+|[0-9]+)).json")
    for f in os.listdir(directory):
        mo = p.match(f)
        if mo is not None:
            sigmas.append(float(mo.group(1)))
    return sorted(sigmas)


def get_stat(func, directory, sigmas, n_iter=None):
    stats = []
    for sigma in sigmas:
        f = os.path.join(directory, "costs_sigma{0:}.json".format(sigma))
        with open(f, "r") as fh:
            costs = json.load(fh)
            if n_iter is not None:
                assert len(costs) >= n_iter
                costs = costs[:n_iter]
        stats.append(func(costs))
    return stats


def plot_config():
    fig, ax = plt.subplots(figsize=(30, 20))
    params = {
        "font.family": "Nimbus Sans L",
        "font.size": 30.0,
        "axes.labelsize": 30.0,
        "axes.titlesize": 30.0,
        "xtick.labelsize": 20.0,
        "ytick.labelsize": 20.0,
    }
    plt.rcParams.update(params)
    plt.grid()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set(linewidth=2)
    ax.spines["bottom"].set(linewidth=2)
    ax.tick_params(axis="both", labelsize=20.0)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return colors


def plot_sigmas(directories, n_iter, save_file=None, errors=True, max_sigma=None):
    colors = plot_config()
    for color, directory in zip(colors, directories):
        print(directory)
        sigmas = find_sigmas(directory)
        print("Sigmas: {0:}".format(sigmas))
        avgs = get_stat(np.mean, directory, sigmas, n_iter)
        print("Averages: {0:}".format(avgs))
        stds = get_stat(np.std, directory, sigmas, n_iter)
        print("Standard dev: {0:}".format(stds))
        amin = np.argmin(avgs)
        print("---- MIN: sigma {0:} - loss {1:}+-{2:}".format(sigmas[amin], avgs[amin], stds[amin]))
        stdse = get_stat(lambda x: np.std(x) / len(x) ** 0.5, directory, sigmas, n_iter)
        plt.plot(sigmas, avgs, linewidth=2.0, color=color, label=os.path.basename(directory),)
        if errors:
            plt.fill_between(
                sigmas,
                np.array(avgs) - np.array(stds),
                np.array(avgs) + np.array(stds),
                alpha=0.3,
                color=color
            )
            plt.fill_between(
                sigmas,
                np.array(avgs) - np.array(stdse),
                np.array(avgs) + np.array(stdse),
                alpha=0.5,
                color=color,
            )

    plt.title(', '.join([os.path.basename(directory) for directory in directories]))
    plt.xlabel("sigma", fontsize=30)
    plt.ylabel("average loss", fontsize=30)
    plt.ylim(bottom=0)
    if max_sigma is not None:
        plt.xlim(right=max_sigma)
    if len(directories) > 1:
        plt.legend()
    if save_file is not None:
        plt.savefig(save_file, transparent=False)
    elif len(directories) == 1:
        plt.savefig(
            os.path.join(directories[0], "avg_loss_over_sigma"), transparent=False,
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="directory with jsons containing costs; plot will be saved here",
        nargs='+'
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="If given only the first n_iter entries in the costs will be used to compute the statistics",
    )
    parser.add_argument(
        "--max_sigma",
        type=float,
        default=None,
        help="Stop sigma-axis at this value."
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help=("If given plot will be saved here. Otherwise plot will only be saved if only a single directory is "
              "given (in that directory).")
    )
    parser.add_argument(
        "--no_errors",
        action='store_true',
        help="If given plot will not contain error bands"
    )
    args = parser.parse_args()
    dirs = [os.path.abspath(dir) for dir in args.directory]
    if args.save_file is not None:
        save_file = os.path.abspath(args.save_file)
    else:
        save_file = None
    plot_sigmas(dirs, args.n_iter, save_file=save_file, errors=not (args.no_errors), max_sigma=args.max_sigma)
