import luigi
import os
import shutil
import z5py
import h5py
import numpy as np
from cremi.io import CremiFile
from find_partners_luigi import FindPartners


def sub(a, b):
    return tuple([a[d] - b[d] for d in range(len(b))])


def add(a, b):
    return tuple([a[d] + b[d] for d in range(len(b))])

class SplitModi(luigi.Task):
    it = luigi.IntParameter()
    path = luigi.Parameter()
    de = luigi.Parameter()
    m = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources = {'ram': 50}
    @property
    def priority(self):
        if int(self.it)%10000==0:
            return 1./int(self.it)
        else:
            return 0.
    
    def requires(self):
        return FindPartners(self.it, self.path, self.de, self.samples, self.data_eval)

    def output(self):
        ret = []
        for s in self.samples:
            ret.append(luigi.LocalTarget(os.path.join(os.path.dirname(self.input()[0].fn),  ('split.{'
                                                                                      '0:}'+self.m+'.msg').format(s))))
        return ret
    def run(self):
        progress = 0.
        self.set_progress_percentage(progress)
        for s in self.samples:
            print(s)
            filename = os.path.join(os.path.dirname(self.input()[0].fn), s+'.h5')
            mask_filename = os.path.join('/groups/saalfeld/saalfeldlab/larissa/data/cremieval', self.de, s+'.n5')
            mask_dataset = 'volumes/masks/'+self.m
            filename_tgt = filename.replace('h5', self.m+'.h5')
            #shutil.copy(filename, filename_tgt)
            f = CremiFile(filename, 'a')
            g = CremiFile(filename_tgt, 'a')
            maskf = z5py.File(mask_filename, use_zarr_format=False)
            mask = maskf[mask_dataset]
            off = mask.attrs['offset']
            res = mask.attrs['resolution']
            mask = np.array(mask[:])
            ann = f.read_annotations()
            shift = sub(ann.offset, off)
            ids = ann.ids()
            rmids = []
            for i in ids:
                t, loc = ann.get_annotation(i)
                vx_idx = (np.array(add(loc, shift)) / res).astype(np.int)
                if not mask[tuple(vx_idx)]:
                    rmids.append(i)
            print(rmids)
            for i in rmids:
                print('removing {0:}'.format(i))
                ann.remove_annotation(i)
            print(ann.comments.keys())
            print(ann.pre_post_partners)
            g.write_annotations(ann)
            progress += 100./len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        for o in self.output():
            done = o.open('w')
            done.close()
