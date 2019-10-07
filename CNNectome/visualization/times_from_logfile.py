import re
import numpy as np
import matplotlib.pyplot as plt
import argparse


def readfile(filename, startline=0):
    f = open(filename, 'r')
    text = f.readlines()
    print("Start reading filename from line {0:}".format(startline))
    text = text[startline:]
    text = ''.join(text)
    return text


def extract_times(text, skip_first=True):
    times = [float(x) for x in re.findall("INFO:root:it\d*: ([\d,.]*)", text)]
    if skip_first:
        times = times[1:]
    return times


def stats(times):
    avg = np.mean(times)
    std = np.std(times)
    print("AVG s/iteration: {0:}".format(avg))
    print("STD s/iteration: {0:}".format(std))


def plot_hist(times, name=None):
    weights = np.ones_like(times)/len(times)
    bins = np.linspace(0, 100, 100)
    plt.hist(times, bins, label=name, density=True, alpha=0.3)


def run(filenames, startlines):
    for f, start in zip(filenames, startlines):
        text = readfile(f, startline=start)
        times = extract_times(text)
        print("RESULTS FOR {0:}".format(f))
        stats(times)
        plot_hist(times, name=f)
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics and plot histograms of times extracted from "
                                                 "logfiles")
    parser.add_argument('files', help='logfile', type=str, nargs='+')
    parser.add_argument('--startlines', help='linenumber from which to start reading the associated logfile',
                        type=int, nargs='+', default=None)
    args = parser.parse_args()
    startlines = args.startlines
    if startlines is None:
        startlines = [0] * len(args.files)
    run(args.files, startlines)


if __name__ == '__main__':
    main()
