import luigi
import os
from cleftreport_luigi import CleftReport
from partnerreport_luigi import PartnerReport


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
    iteration = luigi.IntParameter(default=180000)


if __name__ == "__main__":
    luigi.run()
