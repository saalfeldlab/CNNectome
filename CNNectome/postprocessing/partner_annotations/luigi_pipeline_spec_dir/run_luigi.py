import luigi
import os
from cleftreport_luigi import CleftReport
from partnerreport_luigi import PartnerReport
from split_modi_luigi import SplitModi


class AllEvaluations(luigi.WrapperTask):
    up_to_iteration = luigi.IntParameter(default=200000)
    iteration_step = luigi.IntParameter(default=10000, significant=False)
    data_train = luigi.TupleParameter(
        default=("data2016-aligned", "data2016-unaligned")
    )
    data_eval = luigi.TupleParameter(default=("data2017-aligned", "data2017-unaligned"))
    augmentation = luigi.TupleParameter(default=("deluxe", "classic", "lite"))
    mode = luigi.TupleParameter(default=("validation", "training"))
    samples = luigi.TupleParameter(default=("A", "B", "C"))

    def requires(self):
        for it in range(
            self.iteration_step,
            self.up_to_iteration + self.iteration_step,
            self.iteration_step,
        ):
            for dt in self.data_train:
                for aug in self.augmentation:
                    for de in self.data_eval:
                        for m in self.mode:
                            # yield CleftReport(it, dt, aug, de, m, self.samples, self.data_eval)
                            if it > 20000:
                                yield PartnerReport(
                                    it, dt, aug, de, m, self.samples, self.data_eval
                                )


class SingleEvaluation(luigi.WrapperTask):
    iteration = luigi.IntParameter(default=186000)
    path = luigi.Parameter(
        default="/nrs/saalfeld/heinrichl/synapses/pre_and_post/pre_and_post-v9.0/run01/"
    )
    data_eval = luigi.TupleParameter(default=("data2016-aligned", "data2016-unaligned"))
    samples = luigi.TupleParameter(default=("A", "B", "C", "A+", "B+", "C+"))

    def requires(self):
        for de in self.data_eval:
            if "A+" in self.samples or "B+" in self.samples or "C+" in self.samples:
                test_samples = []
                for s in self.samples:
                    if "+" in s:
                        test_samples.append(s)
                test_samples = tuple(test_samples)
                yield SplitModi(
                    self.iteration,
                    self.path,
                    de,
                    "groundtruth",
                    test_samples,
                    self.data_eval,
                )
            if "A" in self.samples or "B" in self.samples or "C" in self.samples:
                training_samples = []
                for s in self.samples:
                    if not ("+" in s):
                        training_samples.append(s)
                training_samples = tuple(training_samples)
                yield PartnerReport(
                    self.iteration,
                    self.path,
                    de,
                    "groundtruth",
                    training_samples,
                    self.data_eval,
                )
                yield PartnerReport(
                    self.iteration,
                    self.path,
                    de,
                    "validation",
                    training_samples,
                    self.data_eval,
                )
                yield PartnerReport(
                    self.iteration,
                    self.path,
                    de,
                    "training",
                    training_samples,
                    self.data_eval,
                )


if __name__ == "__main__":
    luigi.run()
