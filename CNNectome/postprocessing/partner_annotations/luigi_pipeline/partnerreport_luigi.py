import luigi
import os
import json
from CNNectome.utils import config_loader
from cremi.io import CremiFile
from cremi.evaluation import SynapticPartners
from split_modi_luigi import SplitModi


class PartnerReport(luigi.Task):
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
        return SplitModi(
            self.it, self.dt, self.aug, self.de, self.m, self.samples, self.data_eval
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                os.path.dirname(self.input().fn), "partners." + self.m + ".json"
            )
        )

    def run(self):
        progress = 0.0
        results = dict()
        self.set_progress_percentage(progress)
        for s in self.samples:
            truth = os.path.join(
                config_loader.get_config()["synapses"]["cremieval_path"],
                self.de,
                s + "." + self.m + ".h5",
            )
            test = os.path.join(
                os.path.dirname(self.input().fn), s + "." + self.m + ".h5"
            )
            truth = CremiFile(truth, "a")
            test = CremiFile(test, "a")
            synaptic_partners_eval = SynapticPartners()
            print(test.read_annotations())
            (
                fscore,
                precision,
                recall,
                fp,
                fn,
                filtered_matches,
            ) = synaptic_partners_eval.fscore(
                test.read_annotations(),
                truth.read_annotations(),
                truth.read_neuron_ids(),
                all_stats=True,
            )
            results[s] = dict()
            results[s]["fscore"] = fscore
            results[s]["precision"] = precision
            results[s]["recall"] = recall
            results[s]["fp"] = fp
            results[s]["fn"] = fn
            results[s]["filtered_matches"] = filtered_matches
            progress += 100.0 / len(self.samples)
            try:
                self.set_progress_percentage(progress)
            except:
                pass
        with self.output().open("w") as done:
            json.dump(results, done)
