from __future__ import print_function
import h5py
import numpy as np


def rewrite_mask(sample):
    print("Processing sample", sample)
    # f = h5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{'
    #              '0:}_padded_20170424.aligned.hdf'.format(sample), 'r')
    # g = h5py.File('/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{'
    #              '0:}_padded_20170424.aligned.0bg.hdf'.format(sample), 'w')
    f = h5py.File(
        "/groups/saalfeld/saalfeldlab/projects/cremi-synaptic-partners/sample_{"
        "0:}_padded_20160501.aligned.hdf".format(sample),
        "r",
    )
    g = h5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2016/sample_{"
        "0:}_padded_20160501.aligned.0bg.hdf".format(sample),
        "w",
    )
    m = h5py.File(
        "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/sample_{"
        "0:}_padded_20170424.aligned.0bg.hdf".format(sample),
        "r",
    )

    clefts = np.array(f["volumes/labels/clefts"])
    print("masks...", end="")
    mask = np.ones(clefts.shape, dtype=np.uint64)
    mask[clefts == 0xFFFFFFFFFFFFFFFF - 1] = 0
    mask[clefts == 0xFFFFFFFFFFFFFFFF - 2] = 0

    # make mask OUTSIDE = 0xffffffffffffffff to 0, everything else to 1, dtype can be uint64 - put this into
    # volumes/masks/groundtruth, attributes?

    neuron_ids = np.array(f["volumes/labels/neuron_ids"])

    mask[neuron_ids == 0xFFFFFFFFFFFFFFFF - 2] = 0
    mask[neuron_ids == 0xFFFFFFFFFFFFFFFF - 1] = 0
    g.create_dataset("volumes/masks/groundtruth", data=mask, chunks=(26, 256, 256))
    g["volumes/masks/groundtruth"].attrs.create(
        "resolution", f["volumes/labels/clefts"].attrs["resolution"]
    )
    g["volumes/masks/groundtruth"].attrs.create(
        "offset", f["volumes/labels/clefts"].attrs["offset"]
    )
    print("done")

    training_mask = np.array(m["volumes/masks/training"])
    g.create_dataset(
        "volumes/masks/training", data=training_mask, chunks=(26, 256, 256)
    )
    g["volumes/masks/training"].attrs.create(
        "resolution", m["volumes/masks/training"].attrs["resolution"]
    )
    g["volumes/masks/training"].attrs.create(
        "offset", m["volumes/masks/training"].attrs["offset"]
    )

    validation_mask = np.array(m["volumes/masks/validation"])
    g.create_dataset(
        "volumes/masks/validation", data=validation_mask, chunks=(26, 256, 256)
    )
    g["volumes/masks/validation"].attrs.create(
        "resolution", m["volumes/masks/validation"].attrs["resolution"]
    )
    g["volumes/masks/validation"].attrs.create(
        "offset", m["volumes/masks/validation"].attrs["offset"]
    )

    print("clefts...", end="")
    clefts[clefts == 0xFFFFFFFFFFFFFFFF - 2] = 0
    clefts[clefts == 0xFFFFFFFFFFFFFFFF - 1] = 0
    clefts[clefts == 0xFFFFFFFFFFFFFFFF] = 0
    g.create_dataset("volumes/labels/clefts", data=clefts, chunks=(26, 256, 256))
    for k, v in f["volumes/labels/clefts"].attrs.iteritems():
        g["volumes/labels/clefts"].attrs.create(k, v)
    print("done")

    print("neuron_ids...", end="")
    neuron_ids[neuron_ids == 0xFFFFFFFFFFFFFFFF - 2] = 0
    neuron_ids[neuron_ids == 0xFFFFFFFFFFFFFFFF - 1] = 0
    neuron_ids[neuron_ids == 0xFFFFFFFFFFFFFFFF] = 0
    g.create_dataset(
        "volumes/labels/neuron_ids", data=neuron_ids, chunks=(26, 256, 256)
    )
    for k, v in f["volumes/labels/neuron_ids"].attrs.iteritems():
        g["volumes/labels/neuron_ids"].attrs.create(k, v)
    print("done")

    print("raw...", end="")
    # copy raw data
    g.create_dataset(
        "volumes/raw", data=np.array(f["volumes/raw"]), chunks=(26, 256, 256)
    )
    for k, v in f["volumes/raw"].attrs.iteritems():
        g["volumes/raw"].attrs.create(k, v)
    print("done")
    print("attributes...", end="")
    for k, v in f.attrs.iteritems():
        g.attrs.create(k, v)

    g.create_group("annotations")
    for k, v in f["annotations"].attrs:
        g["annotations"].attrs.create(k, v)

    g.create_group("annotations/comments")
    for k, v in f["annotations/comments"].attrs:
        g["annotations/comments"].attrs.create(k, v)
    g.create_dataset(
        "annotations/comments/comments",
        data=f["annotations/comments/comments"],
        chunks=f["annotations/comments/comments"].chunks,
    )
    for k, v in f["annotations/comments/comments"].attrs.iteritems():
        g["annotations/comments/comments"].attrs.create(k, v)
    g.create_dataset(
        "annotations/comments/target_ids",
        data=f["annotations/comments/target_ids"],
        chunks=f["annotations/comments/target_ids"].chunks,
    )
    for k, v in f["annotations/comments/target_ids"].attrs.iteritems():
        g["annotations/comments/target_ids"].attrs.create(k, v)

    g.create_group("annotations/presynaptic_site")
    for k, v in f["annotations/presynaptic_site"].attrs:
        g["annotations/presynaptic_site"].attrs.create(k, v)

    g.create_dataset(
        "annotations/presynaptic_site/partners",
        data=f["annotations/presynaptic_site/partners"],
    )

    for k, v in f["annotations/presynaptic_site/partners"].attrs.iteritems():
        g["annotations/presynaptic_site/partners"].attrs.create(k, v)

    g.create_dataset(
        "annotations/ids", data=f["annotations/ids"], chunks=f["annotations/ids"].chunks
    )
    for k, v in f["annotations/ids"].attrs.iteritems():
        g["annotations/ids"].attrs.create(k, v)

    g.create_dataset(
        "annotations/locations",
        data=f["annotations/locations"],
        chunks=f["annotations/locations"].chunks,
    )
    for k, v in f["annotations/locations"].attrs.iteritems():
        g["annotations/locations"].attrs.create(k, v)

    g.create_dataset(
        "annotations/types",
        data=f["annotations/types"],
        chunks=f["annotations/types"].chunks,
    )
    for k, v in f["annotations/types"].attrs.iteritems():
        g["annotations/types"].attrs.create(k, v)
    print("done")
    f.close()
    g.close()


if __name__ == "__main__":
    for sample in ["A", "B", "C"]:  # , 'B', 'C']:
        rewrite_mask(sample)
