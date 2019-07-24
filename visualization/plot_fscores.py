import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
import csv


def plot_param(param_settings, plot_by_field="seg", fields_to_plot="fscore 0"):
    plot_dict = dict()
    for field_to_plot in fields_to_plot:
        plot_dict[field_to_plot] = dict()
    for param_setting in param_settings:
        with open("/groups/saalfeld/home/heinrichl/Downloads/prepost.csv", "r") as f:
            f.readline()
            csv_reader = csv.DictReader(f, delimiter=",")
            fields = csv_reader.fieldnames

            for row in csv_reader:
                # print(row['name'])
                found = True
                for k, i in param_setting.iteritems():
                    try:
                        if row[k] != str(i):
                            found = False
                            break
                    except KeyError:
                        continue
                if found:
                    for field_to_plot in fields_to_plot:
                        try:
                            plot_dict[field_to_plot][row[plot_by_field]].append(
                                row[field_to_plot]
                            )
                        except KeyError:
                            plot_dict[field_to_plot][row[plot_by_field]] = [
                                row[field_to_plot]
                            ]
    print(plot_dict)
    numplots_per_param_sett = len(fields_to_plot)
    dist = 1.0 / (numplots_per_param_sett + 1)
    # fig, ax = plt.subplots()
    plt.ylabel("fscore")
    print(dist)
    labels = [ps["legend"] for ps in param_settings]
    for field_idx, field_to_plot in enumerate(fields_to_plot):
        pd = plot_dict[field_to_plot]
        for k, i in pd.iteritems():
            print([float(x) for x in i])
            if field_idx == 0:
                plt.plot(
                    [(field_idx + 1) * dist + x - 0.5 for x in xrange(0, len(i))],
                    [float(x) for x in i],
                    "o",
                    label=k,
                )
            else:
                plt.plot(
                    [(field_idx + 1) * dist + x - 0.5 for x in xrange(0, len(i))],
                    [float(x) for x in i],
                    "o",
                )
        plt.gca().set_color_cycle(None)
        # plt.gca().set_xticks([(field_idx+1)*dist+x-0.5 for x in xrange(0, len(i))])
        plt.xticks([])
    plt.legend()
    plt.xticks(xrange(0, len(i)), labels)
    plt.gca().xaxis.set_ticks_position("none")

    # plt.xticks(xrange(0, len(i)), labels, minor=True)
    # labels = [item.get_text() for item in plt.gca().get_xticklables()]

    plt.ylim([0.5, 1])


param_settings = [
    {
        "iterations": 400000,
        "pre thr": "-",
        "size thr": 50,
        "cleft thr": 127,
        "sample": "C",
        "legend": "baseline",
    },
    {
        "iterations": 400000,
        "pre thr": 35,
        "size thr": 50,
        "cleft thr": 127,
        "sample": "C",
        "legend": "params1",
    },
    {
        "iterations": 400000,
        "pre thr": 60,
        "size thr": 50,
        "cleft thr": 127,
        "sample": "C",
        "legend": "params2",
    },
]
if __name__ == "__main__":

    plot_param(param_settings, fields_to_plot=["fscore 0", "fscore 2"])
    plt.show()
