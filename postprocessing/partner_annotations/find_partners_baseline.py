from __future__ import print_function
import z5py
import os
import numpy as np
import numpy.ma as ma
import scipy.ndimage
import itertools
import cremi
import sys
from joblib import Parallel, delayed
import multiprocessing

SEG_BG_VAL = 0


def bbox_ND(img):
    N = img.ndim
    out = []
    for ax in list(itertools.combinations(range(N), N - 1))[::-1]:
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


class SynapticRegion(object):
    def __init__(self, segmentid, parentcleft, dist_thr=90):
        self.segmentid = segmentid
        self.cleft = parentcleft
        self.dist_thr = dist_thr
        self.erosion_steps = parentcleft.dilation_steps
        self.intersect_with_dilated_cleft_mask = True
        self.distances = []
        self.size = None
        self.pre_status = None
        self.post_status = None

        # these are wasteful to keep
        self.region_for_acc = None
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
        if self.region_for_acc is None:
            self.make_region_for_acc()
        return self.region_for_acc

    def make_region_for_acc(self):
        self.region_for_acc = np.copy(self.cleft.get_cleft_mask())
        self.region_for_acc[
            np.logical_not(self.cleft.get_seg() == self.segmentid)
        ] = False

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

    def set_post_status(self, post_status):
        self.post_status = post_status

    def set_pre_status(self, pre_status):
        self.pre_status = pre_status

    def is_pre(self):
        return self.pre_status

    def is_post(self):
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

    def get_region_for_point(self):
        if self.region_for_point is None:
            self.make_region_for_point()
        return self.region_for_point

    def make_region_for_point(self):
        if self.intersect_with_dilated_cleft_mask:
            self.region_for_point = np.logical_and(
                self.get_segmask_eroded(), self.cleft.get_dilated_cleft_mask()
            )
            if not np.any(self.region_for_point):
                print(
                    "After intersection, no region left for: ",
                    self.segmentid,
                    self.cleft.cleft_id,
                )
                self.region_for_point = self.get_segmask_eroded()
        else:
            self.region_for_point = self.get_segmask_eroded()

    def get_distance_map(self):
        if self.distance_map is None:
            self.compute_distance_map()
        return self.distance_map

    def compute_distance_map(self):
        self.distance_map = scipy.ndimage.morphology.distance_transform_edt(
            np.logical_not(self.get_region_for_point()), sampling=(40, 4, 4)
        )

    def partner_with_post(self, partner):
        if self == partner:
            return None
        assert self.is_pre()
        assert partner.is_post()
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
        return pre_spot, post_spot


class Cleft(object):
    def __init__(
        self, matchmaker, cleft_id, dilation_steps=7, safe_mem=False, size_thr=50
    ):
        self.mm = matchmaker
        self.cleft_id = cleft_id
        self.safe_mem = safe_mem
        self.size_thr = size_thr
        cleft_mask_full = self.mm.cleft_cc_np == cleft_id

        bbox = bbox_ND(cleft_mask_full)

        bbox = [
            bb + shift
            for bb, shift in zip(
                bbox,
                [
                    -(3 * dilation_steps) // 10,
                    1 + (3 * dilation_steps) // 10,
                    -3 * dilation_steps,
                    3 * dilation_steps + 1,
                    -3 * dilation_steps,
                    3 * dilation_steps + 1,
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
        if self.safe_mem:
            self.cleft_mask = None
        else:
            self.cleft_mask = cleft_mask_full[self.bbox_slice]
        del cleft_mask_full
        self.dilated_cleft_mask = None
        self.dilation_steps = dilation_steps
        self.synregions = [
            SynapticRegion(segid, self) for segid in self.find_segments()
        ]

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
        if len(self.synregions) == 0:
            return []
        pre_synregs = []
        post_synregs = []
        partners = []
        sizes = []

        for synreg in self.synregions:
            sizes.append(synreg.get_size())
        try:
            pre_idx = np.argmax(sizes)
        except ValueError:
            print(sizes)
            for synreg in self.synregions:
                print("id", synreg.segmentid)
                print("reg", synreg.get_region_for_acc())
        self.synregions[pre_idx].set_pre_status(True)
        for synreg, size in zip(self.synregions, sizes):
            if synreg.is_pre():
                pre_synregs.append(synreg)
            else:
                if size > self.size_thr:
                    synreg.set_post_status(True)
                    post_synregs.append(synreg)
        for pre in pre_synregs:
            for post in post_synregs:
                answer = pre.partner_with_post(post)
                if answer is None or not answer:
                    continue
                pre_loc, post_loc = answer
                pre_loc = (cpl + bboff for cpl, bboff in zip(pre_loc, self.bbox[::2]))
                post_loc = (cpl + bboff for cpl, bboff in zip(post_loc, self.bbox[::2]))
                partners.append((pre_loc, post_loc, pre.get_size(), post.get_size()))
        return partners

    def uninitialize_mem_save(self):
        for synreg in self.synregions:
            synreg.uninitialize_mem_save()
        self.dilated_cleft_mask = None
        self.seg = None

        if self.safe_mem:
            self.cleft_mask = None


class Matchmaker(object):
    def __init__(
        self,
        syn_file,
        cleft_cc_ds,
        seg_file,
        seg_ds,
        tgt_file,
        raw_file=None,
        raw_ds=None,
        offset=(0.0, 0.0, 0.0),
        num_cores=10,
        safe_mem=False,
    ):
        self.synf = z5py.File(syn_file, use_zarr_format=False)
        self.segf = z5py.File(seg_file, use_zarr_format=False)
        self.cleft_cc = self.synf[cleft_cc_ds]
        self.cleft_cc_np = self.synf[cleft_cc_ds][:]
        self.seg = self.segf[seg_ds]
        self.partners = None
        self.num_cores = num_cores
        # inputs = np.unique(self.cleft_cc[:])[1:]
        # self.list_of_clefts = Parallel(n_jobs=self.num_cores)(delayed(Cleft.__init__)(Cleft.__new__(Cleft), self,
        # cid) for cid in inputs)
        print("finding all clefts...")
        try:
            self.list_of_cleftids = range(1, self.cleft_cc.attrs["max_id"] + 1)
        except AssertionError:
            self.list_of_cleftids = np.unique(self.cleft_cc[:])[1:]
        self.list_of_clefts = [
            Cleft(self, cid, safe_mem=safe_mem) for cid in self.list_of_cleftids
        ]
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
                self.cleft_cc[:], resolution=(40.0, 4.0, 4.0), offset=self.offset
            )
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
        self, distances_filename, presizes_filename, postsizes_filename, sizes_filename
    ):
        distances = []
        presizes = []
        postsizes = []
        sizes = []
        for cleft in self.list_of_clefts:
            for synr in cleft.synregions:
                sizes.append(synr.size)
                if synr.is_pre():
                    presizes.append(synr.size)
                if synr.is_post():
                    postsizes.append(synr.size)
                distances.extend(synr.distances)

        fmt = "%.5g"
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
            annotations.add_comment(preid, ", size: " + str(partner[2]))
            postid = syncounter.next()
            annotations.add_annotation(
                postid,
                "postsynaptic_site",
                tuple(p * r for p, r in zip(partner[1], (40.0, 4.0, 4.0))),
            )
            annotations.add_comment(postid, "size: " + str(partner[3]))
            annotations.set_pre_post_partners(preid, postid)
        self.cremi_file.write_annotations(annotations)


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
#         filename_tgt = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{' \
#                        '0:}_predictions_it400000_accnotdilated_sizefiltered750_twocrit_thr153.hdf'.format(sample)
#         syn_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5'.format(sample)
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
#         mm.extract_dat('/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
#                        '0:}_preness_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
#                        '0:}_postness_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
#                        '0:}_distances_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
#                        '0:}_presizes_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
#                        '0:}_postsizes_accnotdilated_sizefiltered750_twocrit_thr153.dat'.format(sample),
#                        '/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{'
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
    offsets = {
        "A+": (37 * 40, 1176 * 4, 955 * 4),
        "B+": (37 * 40, 1076 * 4, 1284 * 4),
        "C+": (37 * 40, 1002 * 4, 1165 * 4),
        "A": (38 * 40, 942 * 4, 951 * 4),
        "B": (37 * 40, 1165 * 4, 1446 * 4),
        "C": (37 * 40, 1032 * 4, 1045 * 4),
    }
    segf_name = {
        "A+": "sample_A+_85_aff_0.8_cf_hq_dq_dm1_mf0.81_sizefiltered750.n5",
        "B+": "sample_B+_median_aff_0.8_cf_hq_dq_dm1_mf0.87_sizefiltered750.n5",
        "C+": "sample_C+_85_aff_0.8_cf_hq_dq_dm1_mf0.75_sizefiltered750.n5",
    }
    for sample in samples:
        filename_tgt = (
            "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{"
            "0:}_predictions_it400000_baseline_thr{1:}_st50_dt90.hdf".format(
                sample, thr
            )
        )
        syn_file = "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{0:}.n5".format(
            sample
        )
        cleft_cc_ds = "predictions_it400000/cleft_dist_cropped_thr{0:}_cc".format(thr)
        if "+" in sample:
            seg_file = os.path.join(
                "/nrs/saalfeld/heinrichl/synapses/pre_and_post/", segf_name[sample]
            )
            seg_ds = "main"
            raw_file = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{0:}_padded_aligned.n5".format(
                sample
            )
            raw_ds = "volumes/raw"
        else:
            seg_file = syn_file
            seg_ds = "volumes/labels/neuron_ids_cropped"
            raw_file = (
                "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{"
                "0:}_padded_20170424.aligned.0bg.n5".format(sample)
            )
            raw_ds = "volumes/raw"
        # seg_file = '/nrs/saalfeld/heinrichl/synapses/pre_and_post/cremi/{0:}.n5'.format(sample)
        # seg_ds  = 'volumes/labels/neuron_ids_mc_sf750_cropped'
        print("initializing Matchmaker for sample {0:}".format(sample))
        mm = Matchmaker(
            syn_file,
            cleft_cc_ds,
            seg_file,
            seg_ds,
            filename_tgt,
            raw_file,
            raw_ds,
            offsets[sample],
            safe_mem=True,
        )
        print("preparing file for sample {0:}".format(sample))
        mm.prepare_file()
        print("finding partners for sample {0:}".format(sample))
        mm.write_partners()
        mm.cremi_file.close()

        mm.extract_dat(
            "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{"
            "0:}_400000_baseline_thr{1:}_st50_dt90.dat".format(sample, thr),
            "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{"
            "0:}_400000_baseline_thr{1:}_st50_dt90.dat".format(sample, thr),
            "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{"
            "0:}_400000_baseline_thr{1:}_st50_dt90.dat".format(sample, thr),
            "/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v3.0/cremi/{"
            "0:}_400000_baseline_thr{1:}_st50_dt90.dat".format(sample, thr),
        )


if __name__ == "__main__":
    s = [v for v in sys.argv[1:]]
    main(s)
