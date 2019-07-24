import csv
import h5py
import numpy as np
import logging
from cremi import Annotations
from cremi.io import CremiFile

BG_IDS = (0, 0xFFFFFFFFFFFFFFFD)


def make_cleft_to_prepostsyn_neuron_id_dict(csv_file):
    cleft_to_pre = dict()
    cleft_to_post = dict()
    f = open(csv_file, "r")
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        if int(row[12]) >= 0:
            try:
                cleft_to_pre[int(row[12])].add(int(row[0]))
            except KeyError:
                cleft_to_pre[int(row[12])] = {int(row[0])}
            try:
                cleft_to_post[int(row[12])].add(int(row[6]))
            except KeyError:
                cleft_to_post[int(row[12])] = {int(row[6])}
    return cleft_to_pre, cleft_to_post


def filter(
    h5filepath, csv_file_src, csv_file_tgt=None, cleft_ds_name="syncleft_dist_thr0.0_cc"
):
    logging.info(
        "Filtering clefts in {0:}/{1:} with {2:}".format(
            h5filepath, cleft_ds_name, csv_file_src
        )
    )
    cf = CremiFile(h5filepath, "r+")
    ann = cf.read_annotations()
    cleft_to_pre, cleft_to_post = make_cleft_to_prepostsyn_neuron_id_dict(csv_file_src)
    cleft_list_verified = cleft_to_pre.keys()
    logging.info("List of verified clefts:\n{0:}".format(cleft_list_verified))
    cleft_ds = np.array(cf.read_volume(cleft_ds_name).data)

    cleft_list_all = list(np.unique(cleft_ds))
    for bg_id in BG_IDS:
        cleft_list_all.remove(bg_id)
    logging.info("List of all clefts:\n{0:}".format(cleft_list_all))
    cleft_list_unmatched = list(set(cleft_list_all) - set(cleft_list_verified))
    logging.info("List of unmatched clefts:\n{0:}".format(cleft_list_unmatched))
    if csv_file_tgt is not None:
        with open(csv_file_tgt, "w") as f:
            writer = csv.writer(f)
            for i in cleft_list_unmatched:
                writer.writerow([i])
    next_id = max(ann.ids()) + 1
    logging.info("Adding annotations...")
    for cleft_id in cleft_list_unmatched:
        logging.info("... for cleft {0:}".format(cleft_id))
        cleft_coords = np.where(cleft_ds == cleft_id)
        cleft_center = (
            40.0 * cleft_coords[0][int(len(cleft_coords[0]) / 2.0)],
            4.0 * cleft_coords[1][int(len(cleft_coords[1]) / 2.0)],
            4.0 * cleft_coords[2][int(len(cleft_coords[2]) / 2.0)],
        )
        ann.add_annotation(next_id, "synapse", cleft_center)
        ann.add_comment(next_id, str(cleft_id))
        next_id += 1
    logging.info("Saving annotations...")
    cf.write_annotations(ann)
    cf.close()
    logging.info("...done \n\n")


def main():
    for sample in ["A", "B", "C"]:
        h5filepath = "/groups/saalfeld/home/heinrichl/sample_{0:}_padded_20181003.aligned.hdf".format(
            sample
        )
        csv_file = "/groups/saalfeld/home/heinrichl/sample_{0:}_partners.csv".format(
            sample
        )
        csv_file_tgt = "/groups/saalfeld/home/heinrichl/sample_{0:}_unmatched_clefts.csv".format(
            sample
        )
        filter(h5filepath, csv_file, csv_file_tgt=csv_file_tgt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
