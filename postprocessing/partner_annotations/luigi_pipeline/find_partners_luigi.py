from __future__ import print_function
import luigi
import z5py
import os
import numpy as np
import numpy.ma as ma
import scipy.ndimage
import itertools
import cremi
from cc_luigi import ConnectedComponents
import logging

SEG_BG_VAL = 0

offsets = dict()
offsets["A"] = {True: (38, 942, 951), False: (38, 911, 911)}
offsets["B"] = {True: (37, 1165, 1446), False: (37, 911, 911)}
offsets["C"] = {True: (37, 1032, 1045), False: (37, 911, 911)}


def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in list(itertools.combinations(range(N), N - 1))[::-1]:
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
    ):
        self.segmentid = segmentid
        self.cleft = parentcleft
        self.pre_thr = pre_thr
        self.post_thr = post_thr
        self.dist_thr = dist_thr
        self.size_thr = size_thr

        self.erosion_steps = parentcleft.dilation_steps
        self.dilation_steps = parentcleft.dilation_steps

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
        return self.region_for_acc

    def get_eroded_region(self):
        if self.eroded_region is None:
            self.erode_region()
        return self.eroded_region

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
                (post_spot * np.array([40, 4, 4]) + f * vec) / (np.array([40, 4, 4]))
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
        pre_thr=42,
        post_thr=42,
        size_thr=5,
        dist_thr=600,
    ):
        self.mm = matchmaker
        self.cleft_id = cleft_id
        self.safe_mem = safe_mem

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
        if self.safe_mem:
            self.cleft_mask = None
        else:
            self.cleft_mask = cleft_mask_full[self.bbox_slice]
        del cleft_mask_full
        self.dilation_steps = dilation_steps
        self.dilated_cleft_mask = None

        # self.region_for_acc = np.copy(self.get_cleft_mask())
        # self.region_for_acc[np.logical_not(self.get_seg() == self.segmentid)] = False
        self.segments_overlapping = self.find_segments()

        self.synregions = []
        structure = np.zeros((3, 3, 3))
        structure[1, :] = np.ones((3, 3))
        structure[:, 1, 1] = np.ones((3,))
        for segid in self.segments_overlapping:
            region = np.copy(self.get_cleft_mask())
            region[np.logical_not(self.get_seg() == segid)] = False
            region = region.astype(np.uint32)
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
                    )
                )

    def get_cleft_mask(self):
        if self.cleft_mask is None:
            self.set_cleft_mask()
        return self.cleft_mask

    def set_cleft_mask(self):
        bbox_cleft = self.mm.cleft_cc[self.bbox_slice]
        self.cleft_mask = bbox_cleft == self.cleft_id

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


class Matchmaker(object):
    def __init__(
        self,
        syn_file,
        cleft_cc_ds,
        pre_ds,
        post_ds,
        seg_file,
        seg_ds,
        tgt_file,
        raw_file=None,
        raw_ds=None,
        offset=(0.0, 0.0, 0.0),
        safe_mem=False,
        pre_thr=42,
        post_thr=42,
        dist_thr=600,
        size_thr=5,
    ):
        logging.debug("initializing matchmaker")
        self.synf = z5py.File(syn_file, use_zarr_format=False)
        self.segf = z5py.File(seg_file, use_zarr_format=False)
        self.cleft_cc = self.synf[cleft_cc_ds]
        self.cleft_cc_np = self.synf[cleft_cc_ds][:]
        self.seg = self.segf[seg_ds]
        self.pre = self.synf[pre_ds]
        self.post = self.synf[post_ds]
        self.partners = None
        logging.debug("finding list of cleftids")
        try:
            self.list_of_cleftids = range(1, self.cleft_cc.attrs["max_id"] + 1)
        except AssertionError:
            self.list_of_cleftids = np.unique(self.cleft_cc[:])[1:]
        logging.debug(
            "list of cleftids from {0:} to {1:}".format(
                np.min(self.list_of_cleftids), np.max(self.list_of_cleftids)
            )
        )
        logging.debug("initializing list of clefts")
        self.list_of_clefts = [
            Cleft(
                self,
                cid,
                safe_mem=safe_mem,
                pre_thr=pre_thr,
                post_thr=post_thr,
                dist_thr=dist_thr,
                size_thr=size_thr,
            )
            for cid in self.list_of_cleftids
        ]
        logging.debug("intitialized clefts")
        self.cremi_file = cremi.CremiFile(tgt_file, "w")
        self.offset = offset
        if raw_file is not None:
            self.rawf = z5py.File(raw_file, use_zarr_format=False)
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
            preid = syncounter.next()
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
            postid = syncounter.next()
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


class FindPartners(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    de = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources = {"ram": 400}
    retry_count = 1

    @property
    def priority(self):
        if int(self.it) % 10000 == 0:
            return 1.0 / int(self.it)
        else:
            return 0.0

    def requires(self):
        return ConnectedComponents(
            self.it, self.dt, self.aug, self.de, self.samples, self.data_eval
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(os.path.dirname(self.input().fn), "partners.msg")
        )

    def run(self):
        logging.debug("Starting to run partner finding")
        progress = 0.0
        self.set_progress_percentage(progress)
        thr = 127
        cc_thr = 42
        pre_thr = 42
        post_thr = 42
        dist_thr = 600
        size_thr = 5
        for s in self.samples:
            logging.debug("Starting with sample {0:}".format(s))
            filename = os.path.join(os.path.dirname(self.input().fn), s + ".h5")
            syn_file = os.path.join(os.path.dirname(self.input().fn), s + ".n5")
            cleft_cc_ds = "clefts_cropped_thr{0:}_cc{1:}".format(thr, cc_thr)
            pre_ds = "pre_dist_cropped"
            post_ds = "post_dist_cropped"
            seg_file = os.path.join(
                "/groups/saalfeld/saalfeldlab/larissa/data/cremieval/",
                self.de,
                s + ".n5",
            )
            seg_ds = "volumes/labels/neuron_ids_constis_slf1_sf750_cropped"
            if "unaligned" in self.de:
                aligned = False
            else:
                aligned = True
            off = tuple(np.array(offsets[s][aligned]) * np.array((40, 4, 4)))
            mm = Matchmaker(
                syn_file,
                cleft_cc_ds,
                pre_ds,
                post_ds,
                seg_file,
                seg_ds,
                filename,
                offset=off,
                safe_mem=True,
                dist_thr=dist_thr,
                size_thr=size_thr,
                pre_thr=pre_thr,
                post_thr=post_thr,
            )
            # mm.prepare_file()
            mm.write_partners()
            mm.cremi_file.close()
            del mm
            progress += 100.0 / len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        done = self.output().open("w")
        done.close()
