from CNNectome.utils.hierarchy import hierarchy
import os
import zarr
import numpy as np

gt_version = "v0003"

def get_label_ids_by_category(crop, category):
    return [l[0] for l in crop['labels'][category]]


def get_all_annotated_label_ids(crop):
    return get_label_ids_by_category(crop, "present_annotated") + get_label_ids_by_category(crop, "absent_annotated")


def get_all_annotated_labelnames(crop):
    annotated_labelnames = []
    annotated_label_ids = get_all_annotated_label_ids(crop)
    for labelname, label in hierarchy.items():
        if label.generic_label is not None:
            specific_labels = list(set(label.labelid) - set(label.generic_label))
            generic_condition = (all(l in annotated_label_ids for l in label.generic_label) or
                                 all(l in annotated_label_ids for l in specific_labels))
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
            generic_condition = (any(l in present_label_ids for l in specific_labels) and
                             all(l in annotated_label_ids for l in specific_labels)) or \
                            (any(l in present_label_ids for l in label.generic_label) and
                             all(l in annotated_label_ids for l in label.generic_label))
        else:
            generic_condition = False

        if ((any(l in present_label_ids for l in label.labelid) and
             all(l in annotated_label_ids for l in label.labelid)) or generic_condition):
            present_labelnames.append(labelname)
    return present_labelnames

def get_offset_and_shape_from_crop(crop):
    n5file = zarr.open(crop["parent"], mode="r")
    label_ds = "volumes/groundtruth/{version:}/Crop{cropno:}/labels/all".format(version=gt_version.lstrip("v"), cropno=crop["number"])
    offset_wc = n5file[label_ds].attrs["offset"][::-1]
    offset = tuple(np.array(offset_wc)/4.)
    shape = tuple(np.array(n5file[label_ds].shape)/2.)
    return offset, shape


def get_data_path(crop, s1):
    # todo: consolidate this with get_output_paths from inference template in a utils function
    basename, n5_filename = os.path.split(crop['parent'])
    _, cell_identifier = os.path.split(basename)
    base_n5_filename, n5 = os.path.splitext(n5_filename)
    if s1:
        output_filename = base_n5_filename + '_s1_it{0:}' + n5
    else:
        output_filename = base_n5_filename + '_it{0:}' + n5
    return os.path.join(cell_identifier, output_filename)


def short_cell_id(crop):
    shorts = {
        '/groups/cosem/cosem/data/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm.n5': "HeLa2",
        '/groups/cosem/cosem/data/HeLa_Cell3_4x4x4nm/HeLa_Cell3_4x4x4nm.n5': "HeLa3",
        '/groups/cosem/cosem/data/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2_4x4x4nm.n5': "Macrophage",
        '/groups/cosem/cosem/data/Jurkat_Cell1_4x4x4nm/Jurkat_Cell1_FS96-Area1_4x4x4nm.n5': "Jurkat"
    }
    return shorts[crop["parent"]]


def check_label_in_crop(label, crop):
    return any(lbl in get_label_ids_by_category(crop, "present_annotated") for lbl in label.labelid)


def get_cell_identifier(crop):
    basename, n5_filename = os.path.split(crop["parent"])
    _, cell_identifier = os.path.split(basename)
    return cell_identifier