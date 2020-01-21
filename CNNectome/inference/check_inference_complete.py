import json
import os
import argparse
import re


def check_completeness_single_job(out_file, job_no, iteration):
    if os.path.exists(os.path.join(out_file, 'list_gpu_{0:}.json'.format(job_no))) and os.path.exists(os.path.join(out_file, 'list_gpu_{0:}_{1:}_processed.txt'.format(job_no, iteration))):
        block_list = os.path.join(out_file, 'list_gpu_{0:}.json'.format(job_no))
        block_list_processed = os.path.join(out_file, 'list_gpu_{0:}_{1:}_processed.txt'.format(job_no, iteration))
        with open(block_list, 'r') as f:
            block_list = json.load(f)
            block_list = {tuple(coo) for coo in block_list}
        with open(block_list_processed, 'r') as f:
            list_as_str = f.read()
        list_as_str_curated = '[' + list_as_str[:list_as_str.rfind(']') + 1] + ']'
        processed_list = json.loads(list_as_str_curated)
        processed_list = {tuple(coo) for coo in processed_list}
        if processed_list < block_list:
            complete = False
        else:
            complete = True
    else:
        complete = False
    return complete


def check_completeness(out_file, iteration):
    completeness = []
    p = re.compile("list_gpu_(\d+).json")
    jobs = []
    if not os.path.exists(out_file):
        print(0)
        return 0
    for f in os.listdir(out_file):
        mo = p.match(f)
        if mo  is not None:
            jobs.append(mo.group(1))
    if len(jobs) < 1:
       print(0)
       return False
    for i in jobs:
        completeness.append(check_completeness_single_job(out_file, i, iteration))
    print(int(all(completeness)))
    return all(completeness)


def get_output_paths(raw_data_path, setup_path, output_path):
    if output_path is None:
        basename, n5_filename = os.path.split(raw_data_path)
        assert n5_filename.endswith('.n5')

        # output directory, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/"
        all_data_dir, cell_identifier = os.path.split(basename)
        output_dir = os.path.join(setup_path, cell_identifier)

        # output file, e.g. "(...)/setup01/HeLa_Cell2_4x4x4nm/HeLa_Cell2_4x4x4nm_it10000.n5"
        base_n5_filename, n5 = os.path.splitext(n5_filename)
        output_filename = base_n5_filename + '_it{0:}'.format(iteration) + n5
        out_file = os.path.join(output_dir, output_filename)
    else:
        assert output_path.endswith('.n5') or output_path.endswith('.n5/')
        output_dir = os.path.abspath(os.path.dirname(output_path))
        out_file = os.path.abspath(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    return output_dir, out_file


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_job", type=int)
    parser.add_argument("n_cpus", type=int)
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("iteration", type=int)
    parser.add_argument("--raw_ds", type=str, default="volumes/raw")
    parser.add_argument("--mask_ds", type=str, default="volumes/masks/foreground")
    parser.add_argument("--setup_path", type=str, default='.')
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--finish_interrupted", type=bool, default=False)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--min_sc", type=float, default=None)
    parser.add_argument("--max_sc", type=float, default=None)
    parser.add_argument("--float_range", type=int, nargs="+", default=(-1, 1))
    parser.add_argument("--safe_scale", type=bool, default=False)
    args = parser.parse_args()
    raw_data_path = args.raw_data_path
    iteration = args.iteration
    n_job = args.n_job
    setup_path = args.setup_path
    output_path = args.output_path
    _, out_file = get_output_paths(raw_data_path, setup_path, output_path)
    check_completeness(out_file, iteration)

