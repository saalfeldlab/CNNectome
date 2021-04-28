import argparse
import os
from CNNectome.validation.organelles import run_evaluation
import sys

sys.path.append(os.getcwd())

def detect_setup():
    dirs = os.getcwd().split(os.path.sep)
    for d in dirs[::-1]:
        if "setup" in d:
            return d
    raise FileNotFoundError("")


def detect_8nm():
    import unet_template as setup_config
    if tuple(setup_config.voxel_size_input) == (8,8,8):
        return True
    else:
        return False


def main(alt_args=None):
    parser = argparse.ArgumentParser("Run from a setup directory to automate running evaluations for a specific "
                                     "iteration")
    parser.add_argument("iteration", type=int)
    args = parser.parse_args(alt_args)
    arg_list = ["--save"]
    arg_list += ["--setup", detect_setup()]
    arg_list += ["--iteration", str(args.iteration)]
    run_evaluation.main(arg_list)
    if detect_8nm():
        run_evaluation.main(arg_list + ["--s1"])


if __name__ == "__main__":
    main()