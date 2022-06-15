from cremi.io import CremiFile
from cremi.evaluation import (
    NeuronIds,
    Clefts,
    SynapticPartners,
    SynapticPartnersMultRec,
    SynapticPartnersMultRecGt,
)
import logging
import sys
import os
from CNNectome.utils import config_loader


def evaluate(test, truth):
    synaptic_partners_eval = SynapticPartners()
    fscore, precision, recall, fp, fn, filtered_matches = synaptic_partners_eval.fscore(
        test.read_annotations(),
        truth.read_annotations(),
        truth.read_neuron_ids(),
        all_stats=True,
    )
    logging.info("\tfscore: " + str(fscore))
    logging.info("\tprecision: " + str(precision))
    logging.info("\trecall: " + str(recall))
    logging.info("\tfp: " + str(fp))
    logging.info("\tfn: " + str(fn))

    return fscore


def evaluate_multrec(test, truth):
    synaptic_partners_eval = SynapticPartnersMultRec()
    fscore, precision, recall, fp, fn, filtered_matches = synaptic_partners_eval.fscore(
        test.read_annotations(),
        truth.read_annotations(),
        truth.read_neuron_ids(),
        all_stats=True,
    )
    logging.info("\tfscore: " + str(fscore))
    logging.info("\tprecision: " + str(precision))
    logging.info("\trecall: " + str(recall))
    logging.info("\tfp: " + str(fp))
    logging.info("\tfn: " + str(fn))

    return fscore


def evaluate_multrecgt(test, truth, add_in_file=False):
    synaptic_partners_eval = SynapticPartnersMultRecGt()
    if add_in_file:
        (
            fscore,
            precision,
            recall,
            fp,
            fn,
            filtered_matches,
            annot,
        ) = synaptic_partners_eval.fscore(
            test.read_annotations(),
            truth.read_annotations(),
            truth.read_neuron_ids(),
            all_stats=True,
            add_in_file=add_in_file,
        )

        test.write_annotations(annot)
    else:
        (
            fscore,
            precision,
            recall,
            fp,
            fn,
            filtered_matches,
        ) = synaptic_partners_eval.fscore(
            test.read_annotations(),
            truth.read_annotations(),
            truth.read_neuron_ids(),
            all_stats=True,
            add_in_file=add_in_file,
        )
    logging.info("\tfscore: " + str(fscore))
    logging.info("\tprecision: " + str(precision))
    logging.info("\trecall: " + str(recall))
    logging.info("\tfp: " + str(fp))
    logging.info("\tfn: " + str(fn))

    return fscore


def main(s, mode=0, data=None):
    # samples = ['A','B', 'C']
    samples = [(s.split("/")[-1]).split("_")[0]]
    for sample in samples:
        logging.info("evaluating synapse predictions for sample {0:}".format(sample))
        truth_fn = os.path.join(
            config_loader.get_config()["synapses"]["cremi17_data_path"],
            "sample_{0:}_padded_20170424.aligned.hdf".format(sample),
        )
        if data is not None:
            logging.info(
                "sample {0:} in mode {1:} using {2:}".format(sample, mode, data)
            )
        if (
            data == "val"
            or data == "validation"
            or data == "VAL"
            or data == "VALIDATION"
        ):
            assert s.endswith(".hdf")
            test = CremiFile(s.replace(".hdf", ".validation.hdf"), "a")
            truth = CremiFile(truth_fn.replace(".hdf", ".validation.hdf"), "a")
        elif (
            data == "train"
            or data == "training"
            or data == "TRAIN"
            or data == "TRAINING"
        ):
            assert s.endswith(".hdf")
            test = CremiFile(s.replace(".hdf", ".training.hdf"), "a")
            truth = CremiFile(truth_fn.replace(".hdf", ".training.hdf"), "a")
        else:
            test = CremiFile(s, "a")
            truth = CremiFile(truth_fn, "a")

        if mode == 0:
            evaluate(test, truth)
        elif mode == 1:
            evaluate_multrecgt(test, truth, add_in_file=True)
        elif mode == 2:
            evaluate_multrecgt(test, truth)


def main_all(s):
    # samples = ['A','B', 'C']
    main(s, 0, "VAL")
    main(s, 2, "VAL")
    main(s, 0, "TRAIN")
    main(s, 2, "TRAIN")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sys.argv[1]
    # m = int(sys.argv[2])
    # d = sys.argv[3]
    # main(s, m, d)
    main_all(s)
