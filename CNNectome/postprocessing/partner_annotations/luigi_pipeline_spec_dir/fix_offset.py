import h5py
import os
import numpy as np

offsets = dict()
offsets["A"] = {True: (38, 942, 951), False: (38, 911, 911)}
offsets["B"] = {True: (37, 1165, 1446), False: (37, 911, 911)}
offsets["C"] = {True: (37, 1032, 1045), False: (37, 911, 911)}
samples = ["A", "B", "C"]
# for it in range(12000, 150000+2000, 2000):
for it in [10000]:
    for dt in (
        "data2016-aligned",
        "data2016-unaligned",
        "data2017-aligned",
        "data2017-unaligned",
    ):
        for aug in ("lite", "classic", "deluxe"):
            for de in (
                "data2016-aligned",
                "data2016-unaligned",
                "data2017-aligned",
                "data2017-unaligned",
            ):
                if os.path.exists(
                    os.path.join(
                        "/nrs/saalfeld/heinrichl/synapses/data_and_augmentations",
                        dt,
                        aug,
                        "evaluation",
                        str(it),
                        de,
                        "partners.msg",
                    )
                ):
                    print(
                        os.path.join(
                            "/nrs/saalfeld/heinrichl/synapses/data_and_augmentations",
                            dt,
                            aug,
                            "evaluation",
                            str(it),
                            de,
                            "partners.msg",
                        )
                    )
                    for s in samples:
                        assert os.path.exists(
                            os.path.join(
                                "/nrs/saalfeld/heinrichl/synapses/data_and_augmentations",
                                dt,
                                aug,
                                "evaluation",
                                str(it),
                                de,
                                s + ".h5",
                            )
                        )
                    for s in samples:
                        f = h5py.File(
                            os.path.join(
                                "/nrs/saalfeld/heinrichl/synapses/data_and_augmentations",
                                dt,
                                aug,
                                "evaluation",
                                str(it),
                                de,
                                s + ".h5",
                            ),
                            "a",
                        )
                        if "unaligned" in de:
                            aligned = False
                        else:
                            aligned = True
                        try:
                            assert (
                                tuple(f["/annotations"].attrs["offset"])
                                == offsets[s][aligned]
                            )
                        except KeyError:
                            print(it, dt, aug, de, s, "failed")
                            continue
                        off = tuple(
                            np.array(offsets[s][aligned]) * np.array((40, 4, 4))
                        )
                        f["annotations"].attrs["offset"] = off
                        f.close()
