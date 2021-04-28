import csv
import os
import h5py
import numpy as np
from CNNectome.utils import config_loader
shift = {"A": 1498, "B": 1940, "C": 10954}


def compare(filepath1, filepath2, targetfile, cleft_id_shift, contained_ids):
    """
    :param filepath1: csv-file that has the corrected cleft associations
    :param filepath2: csv-file that should be corrected
    :param targetfile: csv-file that should be created
    :param cleft_id_shift
    :return:
    """

    file1 = open(filepath1, "r")
    file2 = open(filepath2, "r")
    target = open(targetfile, "w")
    reader1 = csv.reader(file1)
    reader2 = csv.reader(file2)
    writer = csv.writer(target)
    lookup_by_coord = dict()
    for row in reader1:
        if row[0] == "pre_label":
            writer.writerow(row)
            next(reader1)
            break

    for row in reader2:
        if row[0] == "pre_label":
            next(reader2)
            break
    for row in reader1:
        pre_coord = (float(row[2]), float(row[3]), float(row[4]))
        post_coord = (float(row[7]), float(row[8]), float(row[9]))
        lookup_by_coord[(pre_coord, post_coord)] = int(row[10])
    print(lookup_by_coord)
    for row in reader2:
        if int(row[10]) == -1:
            pre_coord = (float(row[2]), float(row[3]), float(row[4]))
            post_coord = (float(row[7]), float(row[8]), float(row[9]))
            try:
                cleft = lookup_by_coord[(pre_coord, post_coord)]
                if cleft != -1:
                    if cleft == 21827 or cleft == 3580:
                        cleft_shifted = -2
                    else:
                        cleft_shifted = cleft - cleft_id_shift
                        if cleft_shifted not in contained_ids:
                            cleft_shifted = -3
                else:
                    cleft_shifted = -1
            except KeyError:
                cleft_shifted = -4
                pass
            writer.writerow(row[:-2] + [cleft_shifted, ""])
        else:
            writer.writerow(row)
    file1.close()
    file2.close()
    target.close()


def all_clefts(cleftfile):
    hf = h5py.File(cleftfile, "r")
    return np.unique(hf["volumes/labels/clefts"][:])


if __name__ == "__main__":
    conf = config_loader.get_config()
    file1 = os.path.join(conf["synapses"]["cremi17_data_path"], "cleft-partners_{0:}_2017.csv")
    file2 = os.path.join(conf["synapses"]["cremi16_data_path"], "cleft-partners-{0:}-20160501.aligned.csv")
    newfile = os.path.join(conf["synapses"]["cremi16_data_path"], "cleft-partners-{0:}-20160501.aligned.corrected.csv")
    clefts = os.path.join(conf["synapses"]["cremi16_data_path"], "sample_{0:}_padded_20160501.aligned.0bg.hdf")

    for sample in ["A", "B", "C"]:
        contained_clefts = all_clefts(clefts.format(sample))
        # contained_clefts=[1,2,3]
        compare(
            file1.format(sample),
            file2.format(sample),
            newfile.format(sample),
            shift[sample],
            contained_clefts,
        )
