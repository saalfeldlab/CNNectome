import sys
import luigi
import os
import re
import zarr
import numcodecs
import json
from concurrent.futures import ProcessPoolExecutor
import subprocess
from prepare_luigi import MakeItFolder, CheckCheckpoint
from simpleference.inference.util import get_offset_lists


def single_inference(data_train, augmentation, data_eval, samples, gpu, iteration):
    subprocess.call(
        [
            "/groups/saalfeld/home/heinrichl/Projects/CNNectome/postprocessing/partner_annotations_luigi"
            "/run_inference"
            ".sh",
            data_train,
            augmentation,
            data_eval,
            samples,
            gpu,
            iteration,
        ]
    )


class Predict(luigi.Task):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()
    samples = luigi.TupleParameter()
    data_eval = luigi.TupleParameter()
    resources = {"gpu": 1, "ram": 10}

    @property
    def priority(self):
        if int(self.it) % 10000 == 0:
            return 1.0 + 1.0 / int(self.it)
        else:
            return 0.0

    def requires(self):
        return MakeItFolder(self.it, self.dt, self.aug, self.data_eval)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(os.path.dirname(self.input().fn), "pred.msg")
        )

    def run(self):

        src = "/groups/saalfeld/saalfeldlab/larissa/data/cremieval/{0:}/{1:}.n5"
        tgt = os.path.join(os.path.dirname(self.input().fn), "{0:}", "{1:}.n5")
        output_shape = (71, 650, 650)
        gpu_list = []
        for i in range(8):
            nvsmi = subprocess.Popen(
                "nvidia-smi -d PIDS -q -i {0:}".format(i),
                shell=True,
                stdout=subprocess.PIPE,
            ).stdout.read()
            if "None" in nvsmi:
                gpu_list.append(i)
        completed = []
        for de in self.data_eval:
            for s in self.samples:
                srcf = zarr.open(src.format(de, s), mode="r")
                shape = srcf["volumes/raw"].shape
                tgtf = zarr.open(tgt.format(de, s), mode="a")
                if not os.path.exists(os.path.join(tgt.format(de, s), "clefts")):
                    tgtf.empty(
                        name="clefts",
                        shape=shape,
                        compressor=numcodecs.GZip(6),
                        dtype="uint8",
                        chunks=output_shape,
                    )
                    completed.append(False)
                else:
                    if self.check_completeness()[0]:
                        completed.append(True)
                    else:
                        completed.append(False)
                if not os.path.exists(os.path.join(tgt.format(de, s), "pre_dist")):
                    tgtf.empty(
                        name="pre_dist",
                        shape=shape,
                        compressor=numcodecs.GZip(6),
                        dtype="uint8",
                        chunks=output_shape,
                    )
                    completed.append(False)
                else:
                    if self.check_completeness()[0]:
                        completed.append(True)
                    else:
                        completed.append(False)

                if not os.path.exists(os.path.join(tgt.format(de, s), "post_dist")):

                    tgtf.empty(
                        name="post_dist",
                        shape=shape,
                        compressor=numcodecs.GZip(6),
                        dtype="uint8",
                        chunks=output_shape,
                    )
                    completed.append(False)
                else:
                    if self.check_completeness()[0]:
                        completed.append(True)
                    else:
                        completed.append(False)
                get_offset_lists(
                    shape, gpu_list, tgt.format(de, s), output_shape=output_shape
                )
        if all(completed):
            self.finish()
            return
        self.submit_inference(self.data_eval, gpu_list)

        reprocess_attempts = 0
        while reprocess_attempts < 4:
            complete, reprocess_list = self.check_completeness(gpu_list)
            if complete:
                self.finish()
                return
            else:
                self.set_status_message(
                    "Reprocessing {0:}, try {1:}".format(
                        list(reprocess_list), reprocess_attempts
                    )
                )
                self.submit_inference(tuple(reprocess_list), gpu_list)
                reprocess_attempts += 1
        if reprocess_attempts >= 4:
            raise AssertionError

    def submit_inference(self, data_eval, gpu_list):
        with ProcessPoolExecutor(max_workers=len(gpu_list)) as pp:
            tasks = [
                pp.submit(
                    single_inference,
                    self.dt,
                    self.aug,
                    json.dumps(list(data_eval)).replace(" ", "").replace('"', '\\"'),
                    json.dumps(list(self.samples)).replace(" ", "").replace('"', '\\"'),
                    str(gpu),
                    str(self.it),
                )
                for gpu in gpu_list
            ]
            result = [t.result() for t in tasks]

    def finish(self):
        done = self.output().open("w")
        done.close()

    def check_completeness(self, gpu_list=None):
        complete = True
        reprocess = set()
        tgt = os.path.join(os.path.dirname(self.input().fn), "{0:}", "{1:}.n5")
        pattern = re.compile("list_gpu_[0-7].json")
        for de in self.data_eval:
            for s in self.samples:
                if gpu_list is None:
                    gpu_list = []
                    for fn in os.listdir(tgt.format(de, s)):
                        if pattern.match(fn) is not None:
                            gpu_list.append(int(list(filter(str.isdigit, fn))))
                if len(gpu_list) == 0:
                    complete = False
                    reprocess.add(de)
                for gpu in gpu_list:
                    if os.path.exists(
                        os.path.join(tgt.format(de, s), "list_gpu_{0:}.json").format(
                            gpu
                        )
                    ) and os.path.exists(
                        os.path.join(
                            tgt.format(de, s), "list_gpu_{0:}processed.txt".format(gpu)
                        )
                    ):
                        block_list = os.path.join(
                            tgt.format(de, s), "list_gpu_{0:}.json"
                        ).format(gpu)
                        block_list_processed = os.path.join(
                            tgt.format(de, s), "list_gpu_{0:}processed.txt".format(gpu)
                        )
                        with open(block_list, "r") as f:
                            block_list = json.load(f)
                            block_list = {tuple(coo) for coo in block_list}
                        with open(block_list_processed, "r") as f:
                            list_as_str = f.read()
                        list_as_str_curated = (
                            "[" + list_as_str[: list_as_str.rfind("]") + 1] + "]"
                        )
                        processed_list = json.loads(list_as_str_curated)
                        processed_list = {tuple(coo) for coo in processed_list}
                        if processed_list < block_list:
                            complete = False
                            reprocess.add(de)
                    else:
                        complete = False
                        reprocess.add(de)
        return complete, reprocess
