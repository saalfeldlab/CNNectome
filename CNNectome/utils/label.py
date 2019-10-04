import collections
import z5py
from gunpowder import ArrayKey


class N5Dataset(object):
    def __init__(
        self,
        filename,
        labeled_voxels,
        special_categories=None,
        data_dir="/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5",
    ):
        self.filename = filename
        self.data_dir = data_dir
        self.full_path = data_dir.format(filename)
        self.labeled_voxels = labeled_voxels
        if special_categories is not None:
            self.special_categories = special_categories
        else:
            self.special_categories = tuple()


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
        # data_dir="/groups/saalfeld/saalfeldlab/larissa/data/cell/{0:}.n5",
        # data_sources=None,
    ):

        self.labelname = labelname
        if not isinstance(labelid, collections.Iterable):
            labelid = (labelid,)
        self.labelid = labelid
        if generic_label is not None and not isinstance(generic_label, collections.Iterable):
            generic_label = (generic_label,)
        self.generic_label = generic_label
        self.targetid = targetid
        self.thr = thr
        self.separate_labelset = separate_labelset
        if self.separate_labelset:
            self.gt_key = ArrayKey("GT_" + self.labelname.upper())
        else:
            self.gt_key = ArrayKey("GT_LABELS")
        self.gt_dist_key = ArrayKey("GT_DIST_" + self.labelname.upper())
        self.pred_dist_key = ArrayKey("PRED_DIST_" + self.labelname.upper())
        self.mask_key = ArrayKey("MASK_" + self.labelname.upper())
        self.scale_loss = scale_loss
        self.add_constant = add_constant
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
            zf = z5py.File(ds.full_path, use_zarr_format=False)
            try:
                for c in zf["volumes/labels/all"].attrs["orig_counts"]:
                    voxels += c
            except KeyError as e:
                print(ds.filename)
                raise e
    return voxels
