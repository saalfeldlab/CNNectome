import luigi
import os
import zarr
import numpy as np
from CNNectome.utils import config_loader
from cremi.io import CremiFile
from find_partners_luigi import FindPartners


def sub(a, b):
    return tuple([a[d] - b[d] for d in range(len(b))])


def add(a, b):
    return tuple([a[d] + b[d] for d in range(len(b))])


class SplitModi(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    de = luigi.Parameter()
    m = luigi.Parameter()
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
        return FindPartners(
            self.it, self.dt, self.aug, self.de, self.samples, self.data_eval
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(os.path.dirname(self.input().fn), "split." + self.m + ".msg")
        )

    def run(self):
        progress = 0.0
        self.set_progress_percentage(progress)
        for s in self.samples:
            print(s)
            filename = os.path.join(os.path.dirname(self.input().fn), s + ".h5")
            mask_filename = os.path.join(
                config_loader.get_config()["synapses"]["cremieval_path"],
                self.de,
                s + ".n5",
            )
            mask_dataset = "volumes/masks/" + self.m
            filename_tgt = filename.replace("h5", self.m + ".h5")
            # shutil.copy(filename, filename_tgt)
            f = CremiFile(filename, "a")
            g = CremiFile(filename_tgt, "a")
            maskf = zarr.open(mask_filename, mode="r")
            mask = maskf[mask_dataset]
            off = mask.attrs["offset"]
            res = mask.attrs["resolution"]
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
                print("removing {0:}".format(i))
                ann.remove_annotation(i)
            print(ann.comments.keys())
            print(ann.pre_post_partners)
            g.write_annotations(ann)
            progress += 100.0 / len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        done = self.output().open("w")
        done.close()
