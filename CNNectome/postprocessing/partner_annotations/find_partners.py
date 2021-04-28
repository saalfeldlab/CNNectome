import zarr
import os
import numpy as np
import numpy.ma as ma
import scipy.ndimage
import itertools
import cremi
import sys
from CNNectome.utils import config_loader
from joblib import Parallel, delayed
import multiprocessing

SEG_BG_VAL = 0


def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in list(itertools.combinations(list(range(N)), N - 1))[::-1]:
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


class SynapticRegion(object):
    def __init__(
        self,
        segmentid,
        parentcleft,
        region,
        pre_thr=42,
        post_thr=42,
        size_thr=5,
        dist_thr=600,
        mvpts=True,
        ngbrs=True,
    ):
        self.segmentid = segmentid
        self.cleft = parentcleft
        self.pre_thr = pre_thr
        self.post_thr = post_thr
        self.dist_thr = dist_thr
        self.size_thr = size_thr
        self.mvpts = mvpts
        self.ngbrs = ngbrs
        self.erosion_steps = parentcleft.dilation_steps
        self.dilation_steps = parentcleft.dilation_steps
        self.intersect_with_dilated_cleft_mask = True
        self.distances = []
        self.pre_evidence = None
        self.size = None
        self.post_evidence = None
        self.pre_status = None
        self.post_status = None

        # these are wasteful to keep
        self.region_for_acc = region
        self.dilated_region = None
        self.eroded_region = None
        self.region_minus_cleft = None
        self.distance_map = None
        self.segmask_eroded = None
        self.region_for_point = None

    def uninitialize_mem_save(self):
        self.region_for_acc = None
        self.region_minus_cleft = None
        self.distance_map = None
        self.segmask_eroded = None
        self.region_for_point = None

    def get_region_for_acc(self):
        # if self.region_for_acc is None:
        #    self.make_region_for_acc()
        # return self.region_for_acc
        return self.region_for_acc

    # def make_region_for_acc(self):
    #    self.region_for_acc = np.copy(self.cleft.get_cleft_mask())
    #    self.region_for_acc[np.logical_not(self.cleft.get_seg() == self.segmentid)] = False

    def get_dilated_region(self):
        if self.dilated_region is None:
            self.dilate_region(self.dilation_steps)
        return self.dilated_region

    def get_eroded_region(self):
        if self.eroded_region is None:
            self.erode_region()
        return self.eroded_region

    def dilate_region(self, steps):
        self.dilated_region = np.copy(self.get_region_for_acc())
        xy_structure = np.zeros((3, 3, 3))
        xy_structure[1, :] = np.ones((3, 3))
        z_structure = np.zeros((3, 3, 3))
        z_structure[:, 1, 1] = np.ones((3,))
        for k in range(int(steps / 10.0)):
            self.dilated_region = scipy.ndimage.morphology.binary_dilation(
                self.dilated_region, structure=xy_structure, iterations=10
            )
            self.dilated_region = scipy.ndimage.morphology.binary_dilation(
                self.dilated_region, structure=z_structure, iterations=1
            )
        if steps % 10 != 0:
            self.dilated_region = scipy.ndimage.morphology.binary_dilation(
                self.dilated_region, structure=xy_structure, iterations=steps % 10
            )

        return self.dilated_region

    def get_region_minus_cleft(self):
        if self.region_minus_cleft is None:
            self.make_region_minus_cleft()
        return self.region_minus_cleft

    def make_region_minus_cleft(self):
        self.region_minus_cleft = self.get_region_for_acc()
        self.region_minus_cleft[self.cleft.get_cleft_mask()] = False

    def get_size(self):
        if self.size is None:
            self.accumulate_acc_size()
        return self.size

    def accumulate_acc_size(self):
        self.size = float(np.sum(self.get_region_for_acc()))

    def get_pre_evidence(self):
        if self.pre_evidence is None:
            self.accumulate_pre_evidence()
        return self.pre_evidence

    def accumulate_pre_evidence(self):
        try:
            ev = (
                np.sum(self.cleft.get_pre()[self.get_region_for_acc()])
                / self.get_size()
            )
        except RuntimeWarning:
            print(np.sum(self.get_region_for_acc()))
            print(self.get_region_for_acc())
            print(self.cleft.get_pre())
            ev = 0
            pass
        print("PRENESS", ev, ev >= self.pre_thr)
        self.pre_evidence = ev
        return ev

    def get_post_evidence(self):
        if self.post_evidence is None:
            self.accumulate_post_evidence()
        return self.post_evidence

    def accumulate_post_evidence(self):
        try:
            ev = (
                np.sum(self.cleft.get_post()[self.get_region_for_acc()])
                / self.get_size()
            )
        except RuntimeWarning:
            print(np.sum(self.get_region_for_acc()))
            print(self.get_region_for_acc())
            print(self.cleft.cleft_id)
            print(self.cleft.get_cleft_mask())
            print(self.cleft.get_dilated_cleft_mask())
            print(self.cleft.get_post())
            ev = 0
            pass
        print("POSTNESS", ev, ev >= self.post_thr)
        self.post_evidence = ev
        return ev

    def is_pre(self):
        if self.pre_status is None:
            self.pre_status = (self.get_pre_evidence() >= self.pre_thr) and (
                self.get_size() >= self.size_thr
            )
        return self.pre_status

    def is_post(self):
        if self.post_status is None:
            self.post_status = (self.get_post_evidence() >= self.post_thr) and (
                self.get_size() >= self.size_thr
            )
        return self.post_status

    def get_segmask_eroded(self):
        if self.segmask_eroded is None:
            self.erode_seg_mask()
        return self.segmask_eroded

    def erode_seg_mask(self):
        self.segmask_eroded = self.cleft.get_seg() == self.segmentid
        xy_structure = np.zeros((3, 3, 3))
        xy_structure[1, :] = np.ones((3, 3))
        z_structure = np.zeros((3, 3, 3))
        z_structure[:, 1, 1] = np.ones((3,))
        for k in range(int(self.erosion_steps / 10.0)):
            self.segmask_eroded = scipy.ndimage.morphology.binary_erosion(
                self.segmask_eroded, structure=xy_structure, iterations=10
            )
            self.segmask_eroded = scipy.ndimage.morphology.binary_erosion(
                self.segmask_eroded, structure=z_structure, iterations=1
            )
        if self.erosion_steps % 10 != 0:
            self.segmask_eroded = scipy.ndimage.morphology.binary_erosion(
                self.segmask_eroded,
                structure=xy_structure,
                iterations=self.erosion_steps % 10,
            )
        if not np.any(self.segmask_eroded):
            self.erosion_steps -= 1
            print(
                "segment {0:} has been eroded so much that it disappeared, try with one less step, i.e. {1:}".format(
                    self.segmentid, self.erosion_steps
                )
            )
            self.erode_seg_mask()

    def erode_region(self):
        self.eroded_region = self.get_region_for_acc()
        xy_structure = np.zeros((3, 3, 3))
        xy_structure[1, :] = np.ones((3, 3))
        z_structure = np.zeros((3, 3, 3))
        z_structure[:, 1, 1] = np.ones((3,))
        for k in range(int(self.erosion_steps / 10.0)):
            self.eroded_region = scipy.ndimage.morphology.binary_erosion(
                self.eroded_region, structure=xy_structure, iterations=10
            )
            self.eroded_region = scipy.ndimage.morphology.binary_erosion(
                self.eroded_region, structure=z_structure, iterations=1
            )
        if self.erosion_steps % 10 != 0:
            self.eroded_region = scipy.ndimage.morphology.binary_erosion(
                self.eroded_region,
                structure=xy_structure,
                iterations=self.erosion_steps % 10,
            )
        if not np.any(self.eroded_region):
            self.erosion_steps -= 1
            print(
                "segment {0:} has been eroded so much that it disappeared, try with one less step, i.e. {1:}".format(
                    self.segmentid, self.erosion_steps
                )
            )
            self.erode_region()

    def get_region_for_point(self):
        if self.region_for_point is None:
            self.make_region_for_point()
        return self.region_for_point

    def make_region_for_point(self):
        if self.intersect_with_dilated_cleft_mask:
            self.region_for_point = np.logical_and(
                self.get_eroded_region(), self.cleft.get_cleft_mask()
            )
            if not np.any(self.region_for_point):
                print(
                    "After intersection, no region left for: ",
                    self.segmentid,
                    self.cleft.cleft_id,
                )
                self.region_for_point = self.get_eroded_region()
        else:
            self.region_for_point = self.get_eroded_region()

    def get_distance_map(self):
        if self.distance_map is None:
            self.compute_distance_map()
        return self.distance_map

    def compute_distance_map(self):
        self.distance_map = scipy.ndimage.morphology.distance_transform_edt(
            np.logical_not(self.get_region_for_point()), sampling=(40, 4, 4)
        )

    def is_neighbor(self, partner):
        structure = np.zeros((3, 3, 3))
        structure[1, :] = np.ones((3, 3))
        structure[:, 1, 1] = np.ones((3,))
        if self.segmentid == partner.segmentid:
            return False
        else:
            neighborregion = (
                self.get_region_for_acc() + partner.get_region_for_acc()
            ).astype(np.uint8)
            num = scipy.ndimage.label(
                neighborregion, output=neighborregion, structure=structure
            )
            del neighborregion
            if num == 1:
                return True
            else:
                return False

    def partner_with_post(self, partner):
        if self == partner:
            return None
        assert self.is_pre()
        assert partner.is_post()
        if self.ngbrs:
            if not self.is_neighbor(partner):
                print(
                    "{0:} and {1:} are not neighbors".format(
                        self.segmentid, partner.segmentid
                    )
                )
                return False
        post_spot = scipy.ndimage.center_of_mass(partner.get_region_for_acc())
        gradient = []
        for gr in partner.parentcleft.get_cleft_gradient():
            gradient.append(
                np.ma.mean(np.ma.array(gr, mask=np.logical_not(partner.region_for_acc)))
            )

        # np.mean(partner.parentcleft.get_cleft())

    def partner_with_post(self, partner):
        if self == partner:
            return None
        assert self.is_pre()
        assert partner.is_post()
        if self.ngbrs:
            if not self.is_neighbor(partner):
                print(
                    "{0:} and {1:} are not neighbors".format(
                        self.segmentid, partner.segmentid
                    )
                )
                return False
        post_masked_distance_map = ma.array(
            self.get_distance_map(), mask=np.logical_not(partner.get_region_for_point())
        )
        post_spot = np.unravel_index(
            np.argmin(post_masked_distance_map), post_masked_distance_map.shape
        )
        post_to_pre_dist = post_masked_distance_map[post_spot]
        self.distances.append(post_to_pre_dist)
        if post_to_pre_dist >= self.dist_thr:
            print(
                "distance {0:} between pre {1:} and post {2:} above threshold".format(
                    post_to_pre_dist, self.segmentid, partner.segmentid
                )
            )
            return False
        pre_masked_distance_map = ma.array(
            partner.get_distance_map(), mask=np.logical_not(self.get_region_for_point())
        )
        pre_spot = np.unravel_index(
            np.argmin(pre_masked_distance_map), pre_masked_distance_map.shape
        )
        if self.mvpts:
            pre_spot = np.array(pre_spot)
            post_spot = np.array(post_spot)
            vec = (post_spot - pre_spot) * np.array([40, 4, 4]) / post_to_pre_dist
            for f in [100, 50, 25, 0]:
                pre_spot_mov = np.round(
                    (pre_spot * np.array([40, 4, 4]) - f * vec) / (np.array([40, 4, 4]))
                ).astype(np.int)
                np.minimum(
                    pre_spot_mov,
                    self.cleft.get_seg().shape - np.array([1, 1, 1]),
                    out=pre_spot_mov,
                )
                np.maximum(pre_spot_mov, [0, 0, 0], out=pre_spot_mov)
                if self.segmentid == self.cleft.get_seg()[tuple(pre_spot_mov)]:
                    pre_spot = pre_spot_mov
                    break

            for f in [100, 50, 25, 0]:
                post_spot_mov = np.round(
                    (post_spot * np.array([40, 4, 4]) + f * vec)
                    / (np.array([40, 4, 4]))
                ).astype(np.int)
                np.minimum(
                    post_spot_mov,
                    partner.cleft.get_seg().shape - np.array([1, 1, 1]),
                    out=post_spot_mov,
                )
                np.maximum(pre_spot_mov, [0, 0, 0], out=post_spot_mov)
                if partner.segmentid == partner.cleft.get_seg()[tuple(post_spot_mov)]:
                    post_spot = post_spot_mov
                    break

            return tuple(pre_spot), tuple(post_spot)


class Cleft(object):
    def __init__(
        self,
        matchmaker,
        cleft_id,
        dilation_steps=7,
        safe_mem=False,
        splitcc=True,
        pre_thr=42,
        post_thr=42,
        size_thr=5,
        dist_thr=600,
        ngbrs=True,
        mvpts=True,
    ):
        self.mm = matchmaker
        self.cleft_id = cleft_id
        self.safe_mem = safe_mem
        self.splitcc = splitcc
        cleft_mask_full = self.mm.cleft_cc_np == cleft_id

        bbox = bbox_ND(cleft_mask_full)

        bbox = [
            bb + shift
            for bb, shift in zip(
                bbox,
                [
                    -(5 * dilation_steps) // 10,
                    1 + (5 * dilation_steps) // 10,
                    -4 * dilation_steps,
                    4 * dilation_steps + 1,
                    -4 * dilation_steps,
                    4 * dilation_steps + 1,
                ],
            )
        ]

        bbox[0] = max(0, bbox[0])
        bbox[1] = min(cleft_mask_full.shape[0], bbox[1])
        bbox[2] = max(0, bbox[2])
        bbox[3] = min(cleft_mask_full.shape[1], bbox[3])
        bbox[4] = max(0, bbox[4])
        bbox[5] = min(cleft_mask_full.shape[2], bbox[5])
        self.bbox = bbox
        self.bbox_slice = (
            slice(bbox[0], bbox[1], None),
            slice(bbox[2], bbox[3], None),
            slice(bbox[4], bbox[5], None),
        )

        self.seg = None
        self.pre = None
        self.post = None
        self.cleft = None
        if self.safe_mem:
            self.cleft_mask = None
        else:
            self.cleft_mask = cleft_mask_full[self.bbox_slice]
        del cleft_mask_full
        self.dilation_steps = dilation_steps
        self.dilated_cleft_mask = None
        self.cleft_gradient = None

        # self.region_for_acc = np.copy(self.get_cleft_mask())
        # self.region_for_acc[np.logical_not(self.get_seg() == self.segmentid)] = False
        self.segments_overlapping = self.find_segments()

        if self.splitcc:
            self.synregions = []
            structure = np.ones((3, 3, 3))
            # structure[1, :] = np.ones((3, 3))
            # structure[:, 1, 1] = np.ones((3,))
            for segid in self.segments_overlapping:
                region = np.copy(self.get_cleft_mask())
                region[np.logical_not(self.get_seg() == segid)] = False
                region = region.astype(np.uint8)
                num = scipy.ndimage.label(region, output=region, structure=structure)
                for k in range(1, num + 1):
                    self.synregions.append(
                        SynapticRegion(
                            segid,
                            self,
                            region == k,
                            pre_thr=pre_thr,
                            post_thr=post_thr,
                            size_thr=size_thr,
                            dist_thr=dist_thr,
                            ngbrs=ngbrs,
                            mvpts=mvpts,
                        )
                    )
        else:
            self.synregions = [
                SynapticRegion(segid, self) for segid in self.segments_overlapping
            ]

    def get_cleft_mask(self):
        if self.cleft_mask is None:
            self.set_cleft_mask()
        return self.cleft_mask

    def set_cleft_mask(self):
        bbox_cleft = self.mm.cleft_cc[self.bbox_slice]
        self.cleft_mask = bbox_cleft == self.cleft_id

    def get_cleft(self):
        if self.cleft is None:
            self.set_cleft()
        return self.cleft

    def set_cleft(self):
        self.cleft = self.mm.cleft[self.bbox_slice]

    def get_cleft_gradient(self):
        if self.cleft_gradient is None:
            self.set_cleft_gradient()
        return self.cleft_gradient

    def set_cleft_gradient(self):
        self.cleft_gradient = np.gradient(self.get_cleft(), [40.0, 4.0, 4.0])

    def get_seg(self):
        if self.seg is None:
            self.set_seg()
        return self.seg

    def set_seg(self):
        self.seg = self.mm.seg[self.bbox_slice]

    def get_pre(self):
        if self.pre is None:
            self.set_pre()
        return self.pre

    def set_pre(self):
        self.pre = self.mm.pre[self.bbox_slice]

    def get_post(self):
        if self.post is None:
            self.set_post()
        return self.post

    def set_post(self):
        self.post = self.mm.post[self.bbox_slice]

    def find_segments(self):
        segments = list(np.unique(self.get_seg()[self.get_cleft_mask()]))
        try:
            segments.remove(SEG_BG_VAL)
        except ValueError:
            pass
        return segments

    def get_dilated_cleft_mask(self):
        if self.dilated_cleft_mask is None:
            self.dilate_cleft_mask(self.dilation_steps)
        return self.dilated_cleft_mask

    def dilate_cleft_mask(self, steps):
        self.dilated_cleft_mask = np.copy(self.get_cleft_mask())
        xy_structure = np.zeros((3, 3, 3))
        xy_structure[1, :] = np.ones((3, 3))
        z_structure = np.zeros((3, 3, 3))
        z_structure[:, 1, 1] = np.ones((3,))
        for k in range(int(steps / 10.0)):
            self.dilated_cleft_mask = scipy.ndimage.morphology.binary_dilation(
                self.dilated_cleft_mask, structure=xy_structure, iterations=10
            )
            self.dilated_cleft_mask = scipy.ndimage.morphology.binary_dilation(
                self.dilated_cleft_mask, structure=z_structure, iterations=1
            )
        if steps % 10 != 0:
            self.dilated_cleft_mask = scipy.ndimage.morphology.binary_dilation(
                self.dilated_cleft_mask, structure=xy_structure, iterations=steps % 10
            )

        return self.dilated_cleft_mask

    def find_all_partners(self):
        pre_synregs = []
        post_synregs = []
        partners = []
        for synreg in self.synregions:
            if synreg.is_pre():
                pre_synregs.append(synreg)
            if synreg.is_post():
                post_synregs.append(synreg)
        for pre in pre_synregs:
            for post in post_synregs:
                answer = pre.partner_with_post(post)
                if answer is None or not answer:
                    continue
                pre_loc, post_loc = answer
                pre_loc = (cpl + bboff for cpl, bboff in zip(pre_loc, self.bbox[::2]))
                post_loc = (cpl + bboff for cpl, bboff in zip(post_loc, self.bbox[::2]))
                partners.append(
                    (
                        pre_loc,
                        post_loc,
                        pre.get_pre_evidence(),
                        pre.get_post_evidence(),
                        pre.get_size(),
                        post.get_pre_evidence(),
                        post.get_post_evidence(),
                        post.get_size(),
                    )
                )
        return partners

    def uninitialize_mem_save(self):
        for synreg in self.synregions:
            synreg.uninitialize_mem_save()
        self.dilated_cleft_mask = None
        self.seg = None
        self.pre = None
        self.post = None
        if self.safe_mem:
            self.cleft_mask = None
            self.cleft = None


class Matchmaker(object):
    def __init__(
        self,
        syn_file,
        cleft_cc_ds,
        cleft_ds,
        pre_ds,
        post_ds,
        seg_file,
        seg_ds,
        tgt_file,
        raw_file=None,
        raw_ds=None,
        offset=(0.0, 0.0, 0.0),
        num_cores=10,
        safe_mem=False,
        pre_thr=42,
        post_thr=42,
        dist_thr=600,
        size_thr=5,
        ngbrs=True,
        mvpts=True,
        splitcc=True,
    ):
        self.synf = zarr.open(syn_file, mode="r")
        self.segf = zarr.open(seg_file, mode="r")
        self.cleft = self.synf[cleft_ds]
        self.cleft_cc = self.synf[cleft_cc_ds]
        self.cleft_cc_np = self.synf[cleft_cc_ds][:]
        self.seg = self.segf[seg_ds]
        self.pre = self.synf[pre_ds]
        self.post = self.synf[post_ds]
        self.partners = None
        self.num_cores = num_cores
        # inputs = np.unique(self.cleft_cc[:])[1:]
        # self.list_of_clefts = Parallel(n_jobs=self.num_cores)(delayed(Cleft.__init__)(Cleft.__new__(Cleft), self,
        # cid) for cid in inputs)
        print("finding all clefts...")
        try:
            self.list_of_cleftids = list(range(1, self.cleft_cc.attrs["max_id"] + 1))
        except AssertionError:
            self.list_of_cleftids = np.unique(self.cleft_cc[:])[1:]
        self.list_of_clefts = [
            Cleft(
                self,
                cid,
                safe_mem=safe_mem,
                splitcc=splitcc,
                pre_thr=pre_thr,
                post_thr=post_thr,
                dist_thr=dist_thr,
                size_thr=size_thr,
                ngbrs=ngbrs,
                mvpts=mvpts,
            )
            for cid in self.list_of_cleftids
        ]
        self.cremi_file = cremi.CremiFile(tgt_file, "w")
        self.offset = offset
        if raw_file is not None:
            self.rawf = zarr.open(raw_file, mode="r")
            self.raw = self.rawf[raw_ds]
        else:
            self.rawf = None
            self.raw = None

    def prepare_file(self):
        if self.raw is not None:
            self.cremi_file.write_raw(
                cremi.Volume(self.raw[:], resolution=(40.0, 4.0, 4.0))
            )

        self.cremi_file.write_neuron_ids(
            cremi.Volume(self.seg[:], resolution=(40.0, 4.0, 4.0), offset=self.offset)
        )
        self.cremi_file.write_clefts(
            cremi.Volume(
                self.cleft_cc_np, resolution=(40.0, 4.0, 4.0), offset=self.offset
            )
        )
        self.cremi_file.write_volume(
            cremi.Volume(self.pre[:], resolution=(40.0, 4.0, 4.0), offset=self.offset),
            "volumes/pre_dist",
            np.uint8,
        )
        self.cremi_file.write_volume(
            cremi.Volume(self.post[:], resolution=(40.0, 4.0, 4.0), offset=self.offset),
            "volumes/post_dist",
            np.uint8,
        )

    def get_partners(self):
        if self.partners is None:
            self.find_all_partners()
        if not self.partners and not self.partners is None:
            print("no partners found")
        return self.partners

    def find_all_partners(self):
        print("finding partners...")
        self.partners = []
        for cleft in self.list_of_clefts:
            self.partners.extend(cleft.find_all_partners())
            cleft.uninitialize_mem_save()

    def extract_dat(
        self,
        preness_filename,
        postness_filename,
        distances_filename,
        presizes_filename,
        postsizes_filename,
        sizes_filename,
    ):
        preness = []
        postness = []
        distances = []
        presizes = []
        postsizes = []
        sizes = []
        for cleft in self.list_of_clefts:
            for synr in cleft.synregions:
                preness.append(synr.pre_evidence)
                postness.append(synr.post_evidence)
                sizes.append(synr.size)
                if synr.is_pre():
                    presizes.append(synr.size)
                if synr.is_post():
                    postsizes.append(synr.size)
                distances.extend(synr.distances)

        fmt = "%.5g"
        np.savetxt(preness_filename, preness, fmt)
        np.savetxt(postness_filename, postness, fmt)
        np.savetxt(distances_filename, distances, fmt)
        np.savetxt(presizes_filename, presizes, fmt)
        np.savetxt(postsizes_filename, postsizes, fmt)
        np.savetxt(sizes_filename, sizes, fmt)

    def write_partners(self):
        annotations = cremi.Annotations(offset=self.offset)
        syncounter = itertools.count(1)
        for partner in self.get_partners():
            preid = next(syncounter)
            annotations.add_annotation(
                preid,
                "presynaptic_site",
                tuple(p * r for p, r in zip(partner[0], (40.0, 4.0, 4.0))),
            )
            annotations.add_comment(
                preid,
                "preness: "
                + str(partner[2])
                + ", postness: "
                + str(partner[3])
                + ", size: "
                + str(partner[4]),
            )
            postid = next(syncounter)
            annotations.add_annotation(
                postid,
                "postsynaptic_site",
                tuple(p * r for p, r in zip(partner[1], (40.0, 4.0, 4.0))),
            )
            annotations.add_comment(
                postid,
                "preness: "
                + str(partner[5])
                + ", postness: "
                + str(partner[6])
                + ", size: "
                + str(partner[7]),
            )
            annotations.set_pre_post_partners(preid, postid)
        self.cremi_file.write_annotations(annotations)


def main_crop():
    samples = ["C+", "A+", "B+"]  # ,'C+','B+']#, 'B+', 'C+']
    offsets = {
        "A+": (37 * 40, 1176 * 4, 955 * 4),
        "B+": (37 * 40, 1076 * 4, 1284 * 4),
        "C+": (37 * 40, 1002 * 4, 1165 * 4),
    }
    offsets = {
        "A+": (37 * 40, 1676 * 4, 1598 * 4),
        "B+": (37 * 40, 2201 * 4, 3294 * 4),
        "C+": (37 * 40, 1702 * 4, 2135 * 4),
    }

    segf_name = {
        "A+": "sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81.n5",
        "B+": "sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87.n5",
        "C+": "sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75.n5",
    }
    for sample in samples:
        setups_path = config_loader.get_config()["synapses"]["training_setups_path"]
        cremi17_data_path = config_loader.get_config()["synapses"]["cremi17_data_path"]
        filename_tgt = os.path.join(setups_path,
                                    "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_predictions_it80000.hdf".format(
                                        sample)
                                    )
        syn_file = os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop.n5".format(
            sample
        ))
        cleft_cc_ds = "predictions_it80000/cleft_dist_cropped_thr127_cc"
        pre_ds = "predictions_it80000/pre_dist_cropped"
        post_ds = "predictions_it80000/post_dist_cropped"
        seg_file = os.path.join(setups_path,
            "pre_and_post/", segf_name[sample]
        )
        seg_ds = "main"
        raw_file = os.path.join(cremi17_data_path, "sample_{0:}_padded_aligned.n5".format(
            sample
        ))
        raw_ds = "volumes/raw"
        print("initializing Matchmaker for sample {0:}".format(sample))
        mm = Matchmaker(
            syn_file,
            cleft_cc_ds,
            pre_ds,
            post_ds,
            syn_file,
            seg_ds,
            filename_tgt,
            raw_file,
            raw_ds,
            offsets[sample],
        )
        print("preparing file for sample {0:}".format(sample))
        mm.prepare_file()
        print("finding partners for sample {0:}".format(sample))
        mm.write_partners()
        mm.extract_dat(
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_preness.dat".format(sample)),
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_postness.dat".format(sample)),
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_distances.dat".format(sample)),
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_presizes.dat".format(sample)),
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_postsizes.dat".format(sample)),
            os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/{0:}_crop_sizes.dat".format(sample)),
        )
        mm.cremi_file.close()


#
# def main_test(samples):
#     #samples = ['A+', 'B+', 'C+']
#     offsets = {
#          'A+': (37*40, 1176*4, 955*4),
#          'B+': (37*40, 1076*4, 1284*4),
#          'C+': (37*40, 1002*4, 1165*4)
#     }
#
#     segf_name = {'A+': 'sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81_sizefiltered750.n5',
#                  'B+': 'sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87_sizefiltered750.n5',
#                  'C+': 'sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75_sizefiltered750.n5',
#                 }
#
#     for sample in samples:
#         filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{' \
#                        '0:}_predictions_it400000_accnotdilated_sizefiltered750_twocrit_thr153.hdf'.format(sample)
#         syn_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{0:}.n5'.format(sample)
#         cleft_cc_ds = 'predictions_it400000/cleft_dist_cropped_thr153_cc'
#         pre_ds = 'predictions_it400000/pre_dist_cropped'
#         post_ds = 'predictions_it400000/post_dist_cropped'
#         seg_file = os.path.join('/nrs/saalfeld/heinrichl/synapses/pre_and_post/', segf_name[sample])
#         seg_ds = 'main'
#         raw_file = '/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{0:}_padded_aligned.n5'.format(sample)
#         raw_ds = 'volumes/raw'
#         print("initializing Matchmaker for sample {0:}".format(sample))
#         mm = Matchmaker(syn_file, cleft_cc_ds, pre_ds, post_ds, seg_file, seg_ds, filename_tgt, raw_file, raw_ds,
#                         offsets[sample])
#         print("preparing file for sample {0:}".format(sample))
#         mm.prepare_file()
#         print("finding partners for sample {0:}".format(sample))
#         mm.write_partners()
#         mm.cremi_file.close()
#
#         mm.extract_dat('/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_preness_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_postness_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_distances_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_presizes_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_postsizes_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/{'
#                        '0:}_sizes_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        )


def main(samples):
    # samples = ['A+', 'B+', 'C+']
    # offsets = {
    #     'A+': (37*40, 1176*4, 955*4),
    #     'B+': (37*40, 1076*4, 1284*4),
    #     'C+': (37*40, 1002*4, 1165*4)
    # }
    thr = 127
    cc_thr = 42
    pre_thr = 42
    post_thr = 42
    dist_thr = 600
    splitcc = True
    ngbrs = True
    mvpts = True
    iteration = 260000
    seg = "constislf1sf750"
    size_thr = 5
    offsets = {
        "A+": (37 * 40, 1176 * 4, 955 * 4),
        "B+": (37 * 40, 1076 * 4, 1284 * 4),
        "C+": (37 * 40, 1002 * 4, 1165 * 4),
        "A": (38 * 40, 942 * 4, 951 * 4),
        "B": (37 * 40, 1165 * 4, 1446 * 4),
        "C": (37 * 40, 1032 * 4, 1045 * 4),
    }
    segf_name = {
        "A+": "sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81",
        "B+": "sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87",
        "C+": "sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75",
    }
    setups_path = config_loader.get_config()["synapses"]["training_setups_path"]
    cremi_path = config_loader.get_config()["synapses"]["cremi17_data_path"]
    for sample in samples:
        path = os.path.join(setups_path, "pre_and_post/pre_and_post-v6.3/cremi/pre_post_accumulated")
        path = os.path.join(path, "it{0:}k".format(iteration // 1000))
        path = os.path.join(path, seg)
        path = os.path.join(path, "thr{0:}_cc{1:}".format(thr, cc_thr))
        path = os.path.join(path, "st{0:}".format(size_thr))
        path = os.path.join(path, "pret{0:}".format(pre_thr))
        path = os.path.join(path, "post{0:}".format(post_thr))
        path = os.path.join(path, "dist{0:}".format(dist_thr))
        if splitcc:
            path = os.path.join(path, "splitcc")
        if ngbrs:
            path = os.path.join(path, "ngbrs")
        if mvpts:
            path = os.path.join(path, "mvpts")
        dir = "{0:}_{1:}k_{2:}_thr{3:}_cc{4:}_st{5:}_pret{6:}_post{7:}".format(
            sample, iteration // 1000, seg, thr, cc_thr, size_thr, pre_thr, post_thr
        )
        if splitcc:
            dir += "_splitcc"
        if ngbrs:
            dir += "_ngbrs"
        if mvpts:
            dir += "_mvpts"
        path = os.path.join(path, dir)
        filename_tgt = (
            "{0:}_predictions_it{1:}_{2:}_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{4:}_st{"
            "5:}_pret{6:}_post{7:}".format(
                sample, iteration, seg, thr, cc_thr, size_thr, pre_thr, post_thr
            )
        )
        if splitcc:
            filename_tgt += "_splitcc"
        if ngbrs:
            filename_tgt += "_ngbrs"
        if mvpts:
            filename_tgt += "_mvpts"
        filename_tgt += ".hdf"
        filename_tgt = os.path.join(path, filename_tgt)

        # path = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v6.3/cremi/pre_post_accumulated/it400k' \
        #        '/gtslf1/' \
        #        'thr{0:}_cc{1:}/st5/pret42/post42/dist600/splitcc/ngbrs/mvpts/{2:}_400k_gtslf1_thr{0:}_cc{' \
        #        '1:}_st5_pret42_post42_splitcc_ngbrs_mvpts'.format(thr, cc_thr, sample)
        # filename_tgt = os.path.join(path,
        #                             '{0:}_predictions_it80000_gtslf1_acccleftnotdilated_regiondilated_twocrit_thr{' \
        #                '1:}_cc{2:}_st5_pre42_post42_splitcc_ngbrs_mvpts.hdf'.format(sample, thr, cc_thr))
        syn_file = os.path.join(setups_path,
                                "pre_and_post/pre_and_post-v6.3/cremi/{0:}.n5".format(sample))
        cleft_cc_ds = "predictions_it{0:}/cleft_dist_cropped_thr{1:}_cc{2:}".format(
            iteration, thr, cc_thr
        )
        pre_ds = "predictions_it{0:}/pre_dist_cropped".format(iteration)
        post_ds = "predictions_it{0:}/post_dist_cropped".format(iteration)

        seg_file = os.path.join(setups_path,
                                "pre_and_post/cremi/{0:}.n5".format(sample))
        if seg == "gtslf1":
            seg_ds = "volumes/labels/neuron_ids_gt_slf1_cropped"
        elif seg == "constislf1sf750":
            seg_ds = "volumes/labels/neuron_ids_constis_slf1_sf750_cropped_masked"
        elif seg == "gt":
            seg_ds = "volumes/labels/neuron_ids_gt_cropped"
        elif seg == "jans":
            seg_file = os.path.join(setups_path,
                "pre_and_post", segf_name[sample] + ".n5"
            )
            seg_ds = "main"
        elif seg == "jans750":
            seg_file = os.path.join(setups_path,"pre_and_post", segf_name[sample] + "_sizefiltered750.n5")
            seg_ds = "main"
        if "+" in sample:
            raw_file = os.path.join(cremi_path, "sample_{0:}_padded_aligned.n5".format(sample))
            raw_ds = "volumes/raw"
        else:
            raw_file = os.path.join(cremi_path, "sample_{0:}_padded_20170424.aligned.0bg.n5".format(sample))
            raw_ds = "volumes/raw"
        print("initializing Matchmaker for sample {0:}".format(sample))
        mm = Matchmaker(
            syn_file,
            cleft_cc_ds,
            pre_ds,
            post_ds,
            seg_file,
            seg_ds,
            filename_tgt,
            raw_file,
            raw_ds,
            offsets[sample],
            safe_mem=True,
            dist_thr=dist_thr,
            size_thr=size_thr,
            pre_thr=pre_thr,
            post_thr=post_thr,
            splitcc=splitcc,
            mvpts=mvpts,
            ngbrs=ngbrs,
        )
        print("preparing file for sample {0:}".format(sample))
        mm.prepare_file()
        print("finding partners for sample {0:}".format(sample))
        mm.write_partners()
        mm.cremi_file.close()
        add = ""
        if ngbrs:
            add += "_ngbrs"
        if mvpts:
            add += "_mvpts"
        mm.extract_dat(
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_preness_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_postness_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_distances_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_presizes_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_postsizes_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
            os.path.join(
                path,
                "{0:}_{1:}_{2:}_sizes_acccleftnotdilated_regiondilated_twocrit_thr{3:}_cc{"
                "4:}_st{5:}_pre{6:}_post{7:}_dist{8:}{9:}.dat".format(
                    sample,
                    iteration,
                    seg,
                    thr,
                    cc_thr,
                    size_thr,
                    pre_thr,
                    post_thr,
                    dist_thr,
                    add,
                ),
            ),
        )


if __name__ == "__main__":
    s = [v for v in sys.argv[1:]]
    main(s)
