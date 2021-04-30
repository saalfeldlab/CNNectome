import pymongo
import argparse
import zarr
import json
import numpy as np
import os
import logging
from CNNectome.utils.label import Label
from CNNectome.utils import config_loader
from CNNectome.utils.cosem_db import MongoCosemDB
from scipy.ndimage.morphology import generate_binary_structure,binary_dilation, binary_erosion, distance_transform_edt
from typing import Any, Dict, List, Optional, Tuple


def distance(labelfield: np.ndarray,
             **kwargs: Any) -> np.ndarray:
    """
    Compute signed distance transform of binary labelfield.

    Args:
        labelfield: Array of binary labels.
        **kwargs: Additional keyword arguments passed on to `scipy.ndimage.morphology.distance_transform_edt`

    Returns:
        Signed distance transform
    """
    inner_distance = distance_transform_edt(binary_erosion(labelfield, border_value=1,
                                                           structure=generate_binary_structure(labelfield.ndim,
                                                                                               labelfield.ndim)),
                                            **kwargs)
    outer_distance = distance_transform_edt(np.logical_not(labelfield), **kwargs)
    result = inner_distance - outer_distance
    return result


def count_with_add(labelfield: np.ndarray,
                   labelid: int,
                   add_constant: int) -> int:
    """
    Count effective frequency of label that is computationally expanded.

    Args:
        labelfield: Array of labels.
        labelid: Id of label to count frequency of.
        add_constant: Constant to add to distances.

    Returns:
        Effective frequency of label in the array.
    """
    binary_labelfield = labelfield == labelid
    distances = distance(binary_labelfield, sampling=(2, 2, 2)) + add_constant
    return np.sum(distances > 0)


def one_crop(crop: Dict[str, Any],
             labels: List[Label],
             gt_version: str = "v0003") -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Calculate the distribution of `labels` for one particular crop.

    Args:
        crop: Instance of an entry in the crop database.
        gt_version: Version of groundtruth annotations, e.g. "v0003"
        labels: List of labels to compute distribution for.

    Returns:
        Dictionaries of number of positive and negative annotations per label in this `crop`.
    """
    data_dir = config_loader.get_config()["organelles"]["data_path"]
    n5file = zarr.open(str(os.path.join(data_dir, crop["parent"])), mode="r")
    blueprint_label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/{{label:}}"
    blueprint_labelmask_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/masks/{{label:}}"
    labelmask_ds = blueprint_labelmask_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    label_ds = blueprint_label_ds.format(version=gt_version.lstrip("v"), cropno=crop["number"])
    pos = dict()
    neg = dict()
    for l in labels:
        pos[l.labelid[0]] = 0
        neg[l.labelid[0]] = 0

    labelfield = np.array(n5file[label_ds.format(label="all")])
    hist, be = np.histogram(labelfield, bins=sorted([l.labelid[0] for l in labels]+[100]))
    counts = dict()
    for b, h in zip(be, hist):
        counts[b] = h

    present_annotated = [ll[0] for ll in crop['labels']["present_annotated"]]
    not_present_annotated = [ll[0] for ll in crop['labels']["absent_annotated"]]

    for ll in present_annotated:
        if ll == 34:
            logging.debug(ll)
        try:
            label = [lli for lli in labels if lli.labelid[0] == ll][0]
        except IndexError:
            continue
        labelmask_ds = labelmask_ds.format(label=label.labelname)
        if labelmask_ds in n5file:
            maskfield = np.array(n5file[labelmask_ds])
            size = np.sum(maskfield)
        else:
            size = crop["dimensions"]["x"]*crop["dimensions"]["y"]*crop["dimensions"]["z"]*8
        if label.separate_labelset:
            sep_labelfield = np.array(n5file[label_ds.format(label=label.labelname)])
            hist, be = np.histogram(sep_labelfield, bins = sorted([l.labelid[0] for l in labels]+[100]))
            counts_separate = dict()
            for b, h in zip(be, hist):
                counts_separate[b] = h
            if label.add_constant is not None and label.add_constant > 0:
                c = count_with_add(sep_labelfield, ll, label.add_constant)
            else:
                c = counts_separate[ll]
            pos[ll] += c
            neg[ll] += size - c

        else:
            if label.add_constant is not None and label.add_constant > 0:
                c = count_with_add(labelfield, ll, label.add_constant)
            else:
                c = counts[ll]
            pos[ll] += c
            neg[ll] += size - c
    for ll in not_present_annotated:
        if ll == 34:
            logging.debug(ll)
        try:
            label = [lli for lli in labels if lli.labelid[0] == ll][0]
        except IndexError:
            continue
        size = crop["dimensions"]["x"] * crop["dimensions"]["y"] * crop["dimensions"]["z"] * 8
        neg[label.labelid[0]] += size
    return pos, neg


def label_dist(labels: List[Label],
               completion_min: int = 6,
               dataset: Optional[str] = None,
               gt_version: str = "v0003",
               save: Optional[str] = None) -> Dict[str, Dict[int, int]]:
    """
    Compute label distribution.

    Args:
        labels: List of labels to compute distribution for.
        completion_min: Minimal completion status for a crop from the database to be included in the distribution.
        dataset: Dataset for which to calculate label distribution. If None calculate across all datasets.
        gt_version: Version of groundtruth for which to accumulate distribution.
        save: File to which to save distributions as json. If None, results won't be saved.

    Returns:
        Dictionary with distributions per label with counts for "positives", "negatives" and the sum of both ("sums").
    """
    db = MongoCosemDB(gt_version=gt_version)
    collection = db.access("crops", db.gt_version)
    db_filter = {"completion": {"$gte": completion_min}}
    if dataset is not None:
        db_filter["dataset_id"] = dataset
    skip = {"_id": 0, "number": 1, "labels": 1, "parent": 1, "dimensions": 1, "dataset_id": 1}
    positives = dict()
    negatives = dict()
    for ll in labels:
        positives[int(ll.labelid[0])] = 0
        negatives[int(ll.labelid[0])] = 0
    for crop in collection.find(db_filter, skip):
        pos, neg = one_crop(crop, labels, db.gt_version)
        for ll, c in pos.items():
            positives[ll] += int(c)
        for ll, c in neg.items():
            negatives[ll] += int(c)

    sums = dict()
    for ll in pos.keys():
        sums[ll] = negatives[ll] + positives[ll]
    stats = dict()
    stats["positives"] = positives
    stats["negatives"] = negatives
    stats["sums"] = sums

    if save is not None:
        if not save.endswith(".json"):
            save += ".json"
        with open(save, "w") as f:
            json.dump(stats, f)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser("Calculate the distribution of labels.")
    parser.add_argument("--dataset", type=str, help=("Dataset id for which to calculate label distribution. If None "
                        "calculate across all datasets."))
    parser.add_argument("--gt_version", type=str, default="v0003", help="Version of groundtruth.")
    parser.add_argument("--save", type=str, help="File to save results to as json", default=".")
    args = parser.parse_args()

    labels = list()
    labels.append(Label("ecs", 1))
    labels.append(Label("plasma_membrane", 2))
    labels.append(Label("mito_lumen", 4))
    labels.append(Label("mito_membrane", 3))
    labels.append(Label("mito_DNA", 5,))
    labels.append(Label("golgi_lumen", 7))
    labels.append(Label("golgi_membrane", 6))
    labels.append(Label("vesicle_lumen",  9))
    labels.append(Label("vesicle_membrane", 8))
    labels.append(Label("MVB_lumen",  11, ))
    labels.append(Label("MVB_membrane", 10))
    labels.append(Label("lysosome_lumen", 13))
    labels.append(Label("lysosome_membrane", 12))
    labels.append(Label("LD_lumen", 15))
    labels.append(Label("LD_membrane", 14))
    labels.append(Label("er_lumen", 17))
    labels.append(Label("er_membrane", 16))
    labels.append(Label("ERES_lumen", 19))
    labels.append(Label("ERES_membrane", 18))
    labels.append(Label("nucleolus", 29, separate_labelset=True))
    labels.append(Label("nucleoplasm", 28))
    labels.append(Label("NE_lumen", 21))
    labels.append(Label("NE_membrane", 20))
    labels.append(Label("nuclear_pore_in", 23))
    labels.append(Label("nuclear_pore_out", 22))
    labels.append(Label("nucleus_generic", 22))
    labels.append(Label("HChrom", 24))
    labels.append(Label("NHChrom", 25))
    labels.append(Label("EChrom", 26))
    labels.append(Label("NEChrom", 27))
    labels.append(Label("microtubules_in",  36))
    labels.append(Label("microtubules_out", 30))
    labels.append(Label("centrosome", 31, add_constant=2, separate_labelset=True))
    labels.append(Label("distal_app", 32))
    labels.append(Label("subdistal_app", 33))
    labels.append(Label("ribosomes", 34, add_constant=8, separate_labelset=True))
    labels.append(Label("nucleus_generic", 37))
    if args.dataset == "None":
        dataset = None
    else:
        dataset = args.dataset
    label_dist(labels, dataset=dataset, gt_version=args.gt_version, save=args.save)


if __name__ == "__main__":
    main()
