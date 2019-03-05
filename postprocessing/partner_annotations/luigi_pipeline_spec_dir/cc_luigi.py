import luigi
import os
import numpy as np
import scipy.ndimage
import z5py
from threshold_luigi import Threshold

class ConnectedComponents(luigi.Task):
    it = luigi.IntParameter()
    path = luigi.Parameter()
    de = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources={'ram': 10}
    @property
    def priority(self):
        if int(self.it)%10000==0:
            return 1./int(self.it)
        else:
            return 0.

    def requires(self):
        return Threshold(self.it, self.path, self.de, self.samples, self.data_eval)

    def output(self):
        return luigi.LocalTarget(os.path.join(os.path.dirname(self.input().fn), 'cc.msg'))

    def run(self):
        thr_high = 127
        thr_low = 42
        dataset_src = 'clefts_cropped_thr{0:}'
        dataset_tgt = 'clefts_cropped_thr{0:}_cc{1:}'.format(thr_high, thr_low)
        progress = 0.
        self.set_progress_percentage(progress)
        for s in self.samples:
            filename = os.path.join(os.path.dirname(self.input().fn), s+'.n5')
            f = z5py.File(filename, use_zarr_format=False)
            assert f[dataset_src.format(thr_high)].attrs['offset'] == f[dataset_src.format(thr_low)].attrs['offset']
            assert f[dataset_src.format(thr_high)].shape == f[dataset_src.format(thr_low)].shape
            f.create_dataset(dataset_tgt,
                             shape=f[dataset_src.format(thr_high)].shape,
                             compression='gzip',
                             dtype='uint64',
                             chunks=f[dataset_src.format(thr_high)].chunks)
            data_high_thr = np.array(f[dataset_src.format(thr_high)][:])
            data_low_thr = np.array(f[dataset_src.format(thr_low)][:])
            tgt = np.ones(data_low_thr.shape, dtype=np.uint64)
            maxid = scipy.ndimage.label(data_low_thr, output=tgt)
            maxes = scipy.ndimage.maximum(data_high_thr, labels=tgt, index=range(1, maxid+1))
            maxes = np.array([0]+list(maxes))
            factors = maxes[tgt]
            tgt *= factors.astype(np.uint64)
            maxid = scipy.ndimage.label(tgt, output=tgt)
            f[dataset_tgt][:] = tgt.astype(np.uint64)
            f[dataset_tgt].attrs['offset'] = f[dataset_src.format(thr_high)].attrs['offset']
            f[dataset_tgt].attrs['max_id'] = maxid
            progress += 100./len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        done = self.output().open('w')
        done.close()
