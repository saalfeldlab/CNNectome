import json
import numpy as np
import matplotlib.pyplot as plt


class ResultFile:
    def __init__(
        self,
        experiment_name,
        sample,
        iteration,
        mode="validation",
        json_name="validation",
    ):
        self.mode = mode
        jsonfile = "synapses/{0:}/{1}.n5/it_{2:}/{3:}.json".format(
            experiment_name, sample, iteration, json_name
        )
        try:
            with open("/nrs/saalfeld/heinrichl/" + jsonfile, "r") as f:
                self.results = json.load(f)
        except IOError:
            with open("/nearline/saalfeld/larissa/" + jsonfile, "r") as f:
                self.results = json.load(f)

    def get_adgt(self):
        if self.mode == "training":
            return self.results["t_dgt"]["mean"]
        elif self.mode == "validation":
            return self.results["v_dgt"]["mean"]
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))

    def get_adf(self):
        if self.mode == "training":
            return self.results["t_df"]["mean"]
        elif self.mode == "validation":
            return self.results["v_df"]["mean"]
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))

    def get_fn(self):
        if self.mode == "training":
            return self.results["t_fn"]
        elif self.mode == "validation":
            return self.results["v_fn"]
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))

    def get_fp(self):
        if self.mode == "training":
            return self.results["t_fp"]
        elif self.mode == "validation":
            return self.results["v_fp"]
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))

    def get_geometric_cremi_score(self):
        if self.mode == "training":
            return (self.results["t_dgt"]["mean"] * self.results["t_df"]["mean"]) ** 0.5
        elif self.mode == "validation":
            return (self.results["v_dgt"]["mean"] * self.results["v_df"]["mean"]) ** 0.5
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))

    def get_cremi_score(self):
        if self.mode == "training":
            return (self.results["t_dgt"]["mean"] + self.results["t_df"]["mean"]) * 0.5
        elif self.mode == "validation":
            return (self.results["v_dgt"]["mean"] + self.results["v_df"]["mean"]) * 0.5
        else:
            raise ValueError("Unknown mode {0:}".format(self.mode))


class ExpResult:
    def __init__(self, experiment_name, all_iterations=(10000, 20000, 30000, 40000)):
        self.experiment_name = experiment_name
        self.all_iterations = all_iterations

    def get_all_it_cs(
        self, mode="validation", json_name="validation", sample=("A", "B", "C")
    ):
        if isinstance(sample, str):
            sample = (sample,)
        res = []
        for it in self.all_iterations:
            res_thisit = []
            for s in sample:
                rf = ResultFile(self.experiment_name, s, it, mode, json_name)
                res_thisit.append(rf.get_cremi_score())
            res.append(np.mean(res_thisit))
        return res

    def get_all_it_gcs(
        self, mode="validation", json_name="validation", sample=("A", "B", "C")
    ):
        if isinstance(sample, str):
            sample = (sample,)
        res = []
        for it in self.all_iterations:
            res_thisit = []
            for s in sample:
                rf = ResultFile(self.experiment_name, s, it, mode, json_name)
                res_thisit.append(rf.get_geometric_cremi_score())
            res.append(np.mean(res_thisit))
        return res

    def get_all_it_adgt(
        self, mode="validation", json_name="validation", sample=("A", "B", "C")
    ):
        if isinstance(sample, str):
            sample = (sample,)
        res = []
        for it in self.all_iterations:
            res_thisit = []
            for s in sample:
                rf = ResultFile(self.experiment_name, s, it, mode, json_name)
                res_thisit.append(rf.get_adgt())
            res.append(np.mean(res_thisit))
        return res

    def get_all_it_adf(
        self, mode="validation", json_name="validation", sample=("A", "B", "C")
    ):
        if isinstance(sample, str):
            sample = (sample,)
        res = []
        for it in self.all_iterations:
            res_thisit = []
            for s in sample:
                rf = ResultFile(self.experiment_name, s, it, mode, json_name)
                res_thisit.append(rf.get_adf())
            res.append(np.mean(res_thisit))
        return res


def smooth(its, x, smoothingstrength=3, ignore_nan=False):
    width = int(np.floor(smoothingstrength / 2))
    new_x = []

    for i, v in enumerate(x[width:-width]):
        avg = []
        for k in range(-width, width + 1):
            if ignore_nan:
                if not np.isnan(x[i + k + 1]):
                    avg.append(x[i + k + 1])
            else:
                avg.append(x[i + k + 1])
        new_x.append(np.mean(avg))

    new_it = its[width:-width]

    return new_it, new_x


def write_dats(experiment_name, samples, add=""):
    iterations = tuple(range(2000, 84000, 2000))
    if experiment_name == "DTU2_Bonly":
        iterations = tuple(range(2000, 56000, 2000))
    er = ExpResult(experiment_name, all_iterations=iterations)
    x = er.get_all_it_cs("validation", sample=samples)
    datfile = "/groups/saalfeld/home/heinrichl/tmp/data/{0:}{1:}{2:}.dat"
    smoothed_it, smoothed_x = smooth(iterations, x)
    shift = int((len(iterations) - len(smoothed_it)) / 2)
    smoothed_x = [" "] * shift + smoothed_x + [" "] * shift
    with open(datfile.format(experiment_name, add, ""), "w") as f:
        for i, v, sv in zip(iterations, x, smoothed_x):
            f.write(str(i) + " " + str(v) + " " + str(sv) + "\n")


def main():
    samples = ("A", "B", "C")
    experiment_names = [
        "baseline_DTU2",
        "DTU2_unbalanced",
        "DTU2-small",
        "DTU2_100tanh",
        "DTU2_150tanh",
        "DTU2_Adouble",
        "baseline_DTU1",
        "DTU1_unbalanced",
        "DTU2_Adouble",
        "DTU2_plus_bdy",
        "DTU1_plus_bdy",
        "BCU2",
        "BCU1",
    ]
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("A", "B", "C"))

    experiment_names = ["DTU2_Adouble", "baseline_DTU2", "baseline_DTU1", "DTU2_Aonly"]
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("A",), add="_onA")

    experiment_names = ["DTU2_Adouble", "baseline_DTU2", "baseline_DTU1"]
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("B",), add="_onB")
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("C",), add="_onC")
    experiment_names = ["DTU2_Bonly"]
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("B",), add="_onB")
    experiment_names = ["DTU2_Conly"]
    for experiment_name in experiment_names:
        write_dats(experiment_name, ("C",), add="_onC")


if __name__ == "__main__":
    # 'DTU2_Aonly',
    # 'DTU2_Bonly',
    # 'DTU2_Conly',
    # main()
    samples = ("A", "B", "C")
    experiment_names = [
        "downsampling_techniques/0309_01",
        "downsampling_techniques/0315_01",
        "downsampling_techniques/0315_02",
        "downsampling_techniques/0316_01",
        "miccai_experiments/baseline_DTU2",
    ]
    for experiment_name in experiment_names:
        if "miccai" in experiment_name:
            iterations = tuple(range(2000, 84000, 2000))
        else:
            iterations = tuple(range(10000, 410000, 10000))
        #    if experiment_name == 'DTU2_Bonly':
        #        iterations = tuple(range(2000, 42000, 2000))
        er = ExpResult(experiment_name, all_iterations=iterations)
        # x = er.get_all_it_adf('validation', sample=samples, json_name='validation_saturated_s100')
        # x2 = er.get_all_it_adgt('validation', sample=samples, json_name='validation_saturated_s100')
        # new_it, new_x = smooth(iterations, x)
        # new_it2, new_x2 = smooth(iterations, x2)
        # plt.subplot(211)
        # plt.plot(new_it, new_x, label=er.experiment_name+'adf')
        # plt.plot(new_it2, new_x2, label=er.experiment_name+'adgt')
        # plt.legend()
        y = er.get_all_it_adf("validation", sample=samples, json_name="validation")
        y2 = er.get_all_it_adgt("validation", sample=samples, json_name="validation")
        y3 = er.get_all_it_gcs("validation", sample=samples, json_name="validation")
        new_it, new_y = smooth(iterations, y)
        new_it2, new_y2 = smooth(iterations, y2)
        new_it3, new_y3 = smooth(iterations, y3)
        # plt.subplot(212)
        # plt.plot(new_it, new_y, label=er.experiment_name+'adf')
        # plt.plot(new_it2, new_y2, label=er.experiment_name+'adgt')
        plt.plot(iterations, y3, label=er.experiment_name)
        plt.legend()
    ##    #y = er.get_all_it_adf('validation', sample=samples)
    ##    #plt.plot(iterations, smooth(y), label='ADF: '+ er.experiment_name)
    ##    #z = er.get_all_it_adgt('validation', sample=samples)
    ##    #plt.plot(iterations,smooth(z), label='ADGT: '+ er.experiment_name)
    #
    plt.show()
    # er = ExpResult('baseline_DTU1', all_iterations=(10000, 20000, 30000))
    # print(er.get_all_it('validation', 'A'))
    ##er = ExpResult('DTU2_150tanh', all_iterations=(10000, 20000, 30000))
    ##print(er.get_all_it('validation', 'A'))
