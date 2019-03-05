import luigi
import os
from cleftreport_luigi import CleftReport
from partnerreport_luigi import PartnerReport
import logging

class AllEvaluations(luigi.WrapperTask):
    it = 90000
    data_train = 'data2016-unaligned'
    augmentation= 'deluxe'
    data_eval = 'data2017-unaligned'
    mode = 'validation'
    samples = luigi.TupleParameter(default=('A', 'B', 'C'))

    def requires(self):
        yield PartnerReport(self.it, self.data_train, self.augmentation, self.data_eval, self.mode, self.samples,
                            (self.data_eval,))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    luigi.run()
