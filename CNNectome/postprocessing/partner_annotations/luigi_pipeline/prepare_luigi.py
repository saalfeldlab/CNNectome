import luigi
import os
from CNNectome.utils import config_loader


class CheckCheckpoint(luigi.ExternalTask):
    it = luigi.IntParameter()
    dt = luigi.Parameter()
    aug = luigi.Parameter()

    @property
    def priority(self):
        if int(self.it) % 10000 == 0:
            return 1.0 / int(self.it)
        else:
            return 0.0

    def output(self):
        base = os.path.join(config_loader.get_config()["synapses"]["training_setups_path"],
            "data_and_augmentations",
            self.dt,
            self.aug,
            "unet_checkpoint_" + str(self.it),
        )
        return [
            luigi.LocalTarget(base + ".data-00000-of-00001"),
            luigi.LocalTarget(base + ".index"),
            luigi.LocalTarget(base + ".meta"),
        ]


class MakeItFolder(luigi.ExternalTask):
    it = luigi.IntParameter()
    dt = luigi.IntParameter()
    aug = luigi.Parameter()
    data_eval = luigi.TupleParameter()

    @property
    def priority(self):
        return self.it

    def requires(self):
        return CheckCheckpoint(self.it, self.dt, self.aug)

    def output(self):
        base = os.path.dirname(self.input()[0].fn)
        return luigi.LocalTarget(
            os.path.join(base, "evaluation", str(self.it), self.data_eval[-1])
        )

    def run(self):
        # make the folders
        base = os.path.dirname(self.input()[0].fn)
        for de in self.data_eval:
            if not os.path.exists(os.path.join(base, "evaluation", str(self.it), de)):
                os.makedirs(os.path.join(base, "evaluation", str(self.it), de))
