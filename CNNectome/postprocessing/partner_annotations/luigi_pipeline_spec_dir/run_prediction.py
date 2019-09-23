import os
import sys

sys.path.append("/groups/saalfeld/home/papec/Work/my_projects/z5/bld27/python")
sys.path.append("/groups/saalfeld/home/heinrichl/Projects/simpleference")
import time
import json
import z5py
from functools import partial
from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.postprocessing import *


def single_gpu_inference(path, data_eval, samples, gpu, iteration):
    weight_meta_graph = os.path.join(path, "unet_checkpoint_{0:}".format(iteration))
    inference_meta_graph = os.path.join(path, "unet_inference")
    net_io_json = os.path.join(path, "net_io_names.json")
    with open(net_io_json, "r") as f:
        net_io_names = json.load(f)

    input_key = net_io_names["raw"]
    output_key = [
        net_io_names["pre_dist"],
        net_io_names["post_dist"],
        net_io_names["cleft_dist"],
    ]
    input_shape = (91, 862, 862)
    output_shape = (71, 650, 650)

    prediction = TensorflowPredict(
        weight_meta_graph,
        inference_meta_graph,
        input_key=input_key,
        output_key=output_key,
    )
    t_predict = time.time()
    for k, de in enumerate(data_eval):
        for s in samples:
            print("{0:} ({1:}/{2:}), {3:}".format(de, k, len(data_eval), s))
            raw_file = "/groups/saalfeld/saalfeldlab/larissa/data/cremieval/{0:}/{1:}.n5".format(
                de, s
            )
            out_file = os.path.join(
                path, "evaluation/{0:}/{1:}/{2:}.n5".format(iteration, de, s)
            )
            offset_file = os.path.join(out_file, "list_gpu_{0:}.json".format(gpu))
            with open(offset_file, "r") as f:
                offset_list = json.load(f)
            run_inference_n5(
                prediction,
                preprocess,
                partial(clip_float_to_uint8, safe_scale=False, float_range=(-1, 1)),
                raw_file,
                out_file,
                offset_list,
                input_shape=input_shape,
                output_shape=output_shape,
                target_keys=("pre_dist", "post_dist", "clefts"),
                input_key="volumes/raw",
                log_processed=os.path.join(
                    out_file, "list_gpu_{0:}processed.txt".format(gpu)
                ),
            )
            t_predict = time.time() - t_predict

            with open(
                os.path.join(
                    os.path.dirname(offset_file),
                    "t-inf_gpu_{0:}_{1:}.txt".format(gpu, iteration),
                ),
                "w",
            ) as f:
                f.write("Inference with gpu %i in %f s\n" % (gpu, t_predict))


if __name__ == "__main__":
    path = sys.argv[1]
    print("run_prediction", sys.argv[2])
    data_eval = json.loads(sys.argv[2])
    samples = json.loads(sys.argv[3])
    gpu = int(sys.argv[4])
    iteration = int(sys.argv[5])
    single_gpu_inference(path, data_eval, samples, gpu, iteration)
