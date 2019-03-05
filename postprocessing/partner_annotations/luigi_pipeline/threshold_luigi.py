import luigi
import os
import numpy as np
import z5py
from crop_luigi import Crop


class Threshold(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    de = luigi.Parameter()
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
        return Crop(self.it, self.dt, self.aug, self.de, self.samples, self.data_eval)
    def output(self):
        return luigi.LocalTarget(os.path.join(os.path.dirname(self.input().fn), 'thr.msg'))
    def run(self):
        thrs = [127,42]
        progress = 0.
        self.set_progress_percentage(progress)
        for s in self.samples:
            filename = os.path.join(os.path.dirname(self.input().fn), s + '.n5')
            dataset_src = 'clefts_cropped'
            dataset_tgt = 'clefts_cropped_thr{0:}'
            f = z5py.File(filename, use_zarr_format=False)
            for t in thrs:
                f.create_dataset(dataset_tgt.format(t),
                                 shape=f[dataset_src].shape,
                                 compression='gzip',
                                 dtype='uint8',
                                 chunks=f[dataset_src].chunks)
                f[dataset_tgt.format(t)][:] = (f[dataset_src][:] > t).astype(np.uint8)
                f[dataset_tgt.format(t)].attrs['offset'] = f[dataset_src].attrs['offset']
            progress += 100./len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        done = self.output().open('w')
        done.close()