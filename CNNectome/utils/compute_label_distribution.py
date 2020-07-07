import pymongo
import zarr
import json
import numpy as np
from CNNectome.utils.label import Label
from scipy.ndimage.morphology import generate_binary_structure,binary_dilation, binary_erosion, distance_transform_edt


def distance(label, **kwargs):
    inner_distance = distance_transform_edt(binary_erosion(label, border_value=1,
                                                           structure=generate_binary_structure(label.ndim,
                                                                                               label.ndim)),
                                            **kwargs)
    outer_distance = distance_transform_edt(np.logical_not(label), **kwargs)
    result = inner_distance - outer_distance
    return result


def count_with_add(labelfield, labelid, add_constant):
    binary_labelfield = labelfield == labelid
    distances = distance(binary_labelfield, sampling=(2, 2, 2)) + add_constant
    return np.sum(distances>0)


def get_label_ids_by_category(crop, category):
    return [l[0] for l in crop['labels'][category]]


def label_filter(cond_f):
    return [ll for ll in labels if cond_f(ll)]

# def count_with_add(labelfield, labelid, add_constant):
#     steps = int(add_constant/2)
#     binary_labelfield = labelfield==labelid
#     return np.sum(binary_dilation(binary_labelfield, generate_binary_structure(3, 1), steps))

def one_crop(crop, gt_version, labels):
    n5file = zarr.open(str(crop["parent"]), mode="r")
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

    present_annotated = get_label_ids_by_category(crop, "present_annotated")
    not_present_annotated = get_label_ids_by_category(crop, "absent_annotated")
    for ll in present_annotated:
        if ll == 34:
            print(ll)
        try:
            label = label_filter(lambda l: l.labelid[0] == ll)[0]
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
            print(ll)
        try:
            label = label_filter(lambda l: l.labelid[0] == ll)[0]
        except IndexError:
            continue
        size = crop["dimensions"]["x"] * crop["dimensions"]["y"] * crop["dimensions"]["z"] * 8
        neg[label.labelid[0]] += size
    return pos, neg


def main(labels, db_username, db_password, db_name="crops", gt_version="v0003", completion_min=6):
    client = pymongo.MongoClient("cosem.int.janelia.org:27017", username=db_username, password=db_password)
    db = client[db_name]
    collection = db[gt_version]
    filter = {"completion": {"$gte": completion_min}}
    skip = {"_id": 0, "number": 1, "labels": 1, "parent": 1, "dimensions": 1}
    positives = dict()
    negatives = dict()
    for l in labels:
        positives[int(l.labelid[0])] = 0
        negatives[int(l.labelid[0])] = 0
    for crop in collection.find(filter, skip):
        print(crop)
        # if int(crop["number"]) in [54,55,56,57,58,59,94,95,96,60,61,62,63,64,65,85,86,87,25,26,81,82,83,84,97,98,99,
        #                            72,73,74,75,76,77,88,89,90,66,67,68,69,70,71,91,92,93]:
        #     print("Skipping {0:}".format(int(crop["number"])))
        #     continue
        # print(crop)
        pos, neg = one_crop(crop, gt_version, labels)
        for l, c in pos.items():
            positives[l] += int(c)
        for l, c in neg.items():
            negatives[l] += int(c)

    sums = dict()
    for l in pos.keys():
        sums[l] = negatives[l]+positives[l]
    print("positives", positives)
    print("negatives", negatives)
    print("sums", sums)
    stats = dict()
    stats["positives"] = positives
    stats["negatives"] = negatives
    stats["sums"] = sums
    with open("stats_new.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":

    db_username = "root"
    db_password = "root"
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
    main(labels, db_username, db_password)