import luigi
import os
import z5py
from prediction_luigi import Predict


offsets = dict()
offsets["A"] = {True: (38, 942, 951), False: (38, 911, 911)}
offsets["B"] = {True: (37, 1165, 1446), False: (37, 911, 911)}
offsets["C"] = {True: (37, 1032, 1045), False: (37, 911, 911)}
shapes = dict()
shapes["A"] = {True: (125, 1438, 1322), False: (125, 1250, 1250)}
shapes["B"] = {True: (125, 1451, 2112), False: (125, 1250, 1250)}
shapes["C"] = {True: (125, 1578, 1469), False: (125, 1250, 1250)}


class Crop(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    de = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources = {"ram": 50}

    @property
    def priority(self):
        if int(self.it) % 10000 == 0:
            return 1.0 / int(self.it)
        else:
            return 0.0

    def requires(self):
        return Predict(self.it, self.dt, self.aug, self.samples, self.data_eval)
        # and so on

    def output(self):
        return luigi.LocalTarget(
            os.path.join(os.path.dirname(self.input().fn), self.de, "crop.msg")
        )

    def run(self):
        progress = 0.0
        self.set_progress_percentage(progress)
        if "unaligned" in self.de:
            aligned = False
        else:
            aligned = True
        for s in self.samples:
            filename = os.path.join(
                os.path.dirname(self.input().fn), self.de, s + ".n5"
            )
            datasets_src = ["clefts", "pre_dist", "post_dist"]
            datasets_tgt = ["clefts_cropped", "pre_dist_cropped", "post_dist_cropped"]
            off = offsets[s][aligned]
            sh = shapes[s][aligned]
            f = z5py.File(filename, use_zarr_format=False)
            for dss, dst in zip(datasets_src, datasets_tgt):
                chunk_size = tuple(min(c, shi) for c, shi in zip(f[dss].chunks, sh))
                f.create_dataset(
                    dst,
                    shape=sh,
                    compression="gzip",
                    dtype=f[dss].dtype,
                    chunks=chunk_size,
                )
                bb = tuple(slice(o, o + shi, None) for o, shi in zip(off, sh))
                f[dst][:] = f[dss][bb]
                f[dst].attrs["offset"] = off[::-1]

                progress += 100.0 / (len(self.samples) * len(datasets_src))
                try:
                    self.set_progress_percentage(progress)
                except:
                    pass

        done = self.output().open("w")
        done.close()
