import collections
import zarr
from gunpowder import ArrayKey


class Label(object):
    def __init__(
        self,
        labelname,
        labelid,
        generic_label=None,
        targetid=None,
        thr=128,
        scale_loss=True,
        scale_key=None,
        add_constant=None,
        separate_labelset=False,
        frac_pos=0.5,
        frac_neg=0.5,
    ):

        self.labelname = labelname
        if not isinstance(labelid, collections.Iterable) and labelid is not None:
            labelid = (labelid,)
        self.labelid = labelid
        if generic_label is not None and not isinstance(
            generic_label, collections.Iterable
        ):
            generic_label = (generic_label,)
        self.generic_label = generic_label
        self.targetid = targetid
        self.thr = thr
        self.separate_labelset = separate_labelset
        if self.separate_labelset and self.separate_labelset is not None:
            self.gt_key = ArrayKey("GT_" + self.labelname.upper())
        elif not self.separate_labelset and self.separate_labelset is not None:
            self.gt_key = ArrayKey("GT_LABELS")
        else:
            self.gt_key = None
        self.gt_dist_key = ArrayKey("GT_DIST_" + self.labelname.upper())
        self.pred_dist_key = ArrayKey("PRED_DIST_" + self.labelname.upper())
        self.mask_key = ArrayKey("MASK_" + self.labelname.upper())
        self.scale_loss = scale_loss
        self.add_constant = add_constant
        self.frac_pos = frac_pos
        self.frac_neg = frac_neg
        # self.data_dir = data_dir
        # self.data_sources = data_sources
        # self.total_voxels = compute_total_voxels(self.data_dir, self.data_sources)
        num = 0
        # if data_sources is not None:
        #     for ds in data_sources:
        #         zf = z5py.File(ds.full_path, use_zarr_format=False)
        #         for l in labelid:
        #             if l in zf["volumes/labels/all"].attrs["relabeled_ids"]:
        #                 num += zf["volumes/labels/all"].attrs["relabeled_counts"][
        #                     zf["volumes/labels/all"].attrs["relabeled_ids"].index(l)
        #                 ]
        # if num > 0:
        #     self.class_weight = float(self.total_voxels) / num
        # else:
        #     self.class_weight = 0.0
        # print(labelname, self.class_weight)

        if self.scale_loss:
            self.scale_key = ArrayKey("SCALE_" + self.labelname.upper())
        if scale_key is not None:
            self.scale_key = scale_key
        if not self.scale_loss and scale_key is None:
            self.scale_key = self.mask_key


def filter_by_category(list_of_datasets, category):
    filtered = []
    for ds in list_of_datasets:
        if category in ds.special_categories:
            filtered.append(ds)
    return filtered


def compute_total_voxels(data_dir, data_sources):
    voxels = 0
    if data_sources is not None:
        for ds in data_sources:
            zf = zarr.open(ds.full_path, mode="r")
            try:
                for c in zf["volumes/labels/all"].attrs["orig_counts"]:
                    voxels += c
            except KeyError as e:
                raise e
    return voxels
