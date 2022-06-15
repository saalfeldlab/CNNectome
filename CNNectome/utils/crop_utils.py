from CNNectome.utils import config_loader
from CNNectome.utils.hierarchy import hierarchy
import os
import zarr
import numpy as np


def get_label_ids_by_category(crop, category):
    return [l[0] for l in crop["labels"][category]]


def get_all_annotated_label_ids(crop):
    return get_label_ids_by_category(
        crop, "present_annotated"
    ) + get_label_ids_by_category(crop, "absent_annotated")


def get_all_annotated_labelnames(crop):
    annotated_labelnames = []
    annotated_label_ids = get_all_annotated_label_ids(crop)
    for labelname, label in hierarchy.items():
        if label.generic_label is not None:
            specific_labels = list(set(label.labelid) - set(label.generic_label))
            generic_condition = all(
                l in annotated_label_ids for l in label.generic_label
            ) or all(l in annotated_label_ids for l in specific_labels)
        else:
            generic_condition = False
        if all(l in annotated_label_ids for l in label.labelid) or generic_condition:
            annotated_labelnames.append(labelname)
    return annotated_labelnames


def get_all_present_labelnames(crop):
    present_labelnames = []
    present_label_ids = get_label_ids_by_category(crop, "present_annotated")
    annotated_label_ids = get_all_annotated_label_ids(crop)
    for labelname, label in hierarchy.items():
        if label.generic_label is not None:
            specific_labels = list(set(label.labelid) - set(label.generic_label))
            generic_condition = (
                any(l in present_label_ids for l in specific_labels)
                and all(l in annotated_label_ids for l in specific_labels)
            ) or (
                any(l in present_label_ids for l in label.generic_label)
                and all(l in annotated_label_ids for l in label.generic_label)
            )
        else:
            generic_condition = False

        if (
            any(l in present_label_ids for l in label.labelid)
            and all(l in annotated_label_ids for l in label.labelid)
        ) or generic_condition:
            present_labelnames.append(labelname)
    return present_labelnames


def get_offset_and_shape_from_crop(crop, gt_version="v0003"):
    n5file = zarr.open(
        os.path.join(
            config_loader.get_config()["organelles"]["data_path"], crop["parent"]
        ),
        mode="r",
    )
    label_ds = "volumes/groundtruth/{version:}/crop{cropno:}/labels/all".format(
        version=gt_version.lstrip("v"), cropno=crop["number"]
    )
    offset_wc = n5file[label_ds].attrs["offset"][::-1]
    offset = tuple(np.array(offset_wc) / 4.0)
    shape = tuple(np.array(n5file[label_ds].shape) / 2.0)
    return offset, shape


def get_data_path(crop, s1):
    # todo: consolidate this with get_output_paths from inference template in a utils function
    cell_id, n5_filename = os.path.split(crop["parent"])
    base_n5_filename, n5 = os.path.splitext(n5_filename)
    if s1:
        output_filename = base_n5_filename + "_s1_it{0:}" + n5
    else:
        output_filename = base_n5_filename + "_it{0:}" + n5
    return os.path.join(cell_id, output_filename)


def alt_short_cell_id(crop):
    shorts = {
        "jrc_hela-2": "HeLa2",
        "jrc_hela-3": "HeLa3",
        "jrc_mac-2": "Macrophage",
        "jrc_jurkat-1": "Jurkat",
    }
    return shorts[crop["dataset_id"]]


def check_label_in_crop(label, crop):
    return any(
        lbl in get_label_ids_by_category(crop, "present_annotated")
        for lbl in label.labelid
    )
