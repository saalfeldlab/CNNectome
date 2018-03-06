import json
import os
processed_configs = []
for gpu in range(8):

    json_file = '/groups/saalfeld/home/heinrichl/Projects/simpleference/experiments/cremi_validation/{' \
                '0:}_processed_configs.json'.format(gpu)
    with open(json_file, 'r') as f:
        processed_configs.append(json.load(f))
experiment_names=[
    'baseline_DTU2',
    'DTU2_unbalanced',
    'DTU2-small',
    'DTU2_100tanh',
    'DTU2_150tanh',
    'DTU2_Aonly',
    'DTU2_Bonly',
    'DTU2_Conly',
    'DTU2_Adouble',
    'baseline_DTU1',
    'DTU1_unbalanced',
    'DTU2_plus_bdy',
    'DTU1_plus_bdy',
    'BCU2',
    'BCU1'
]
for experiment_name in experiment_names:
    if experiment_name != 'DTU2_Bonly':
        iteration = 68000
    else:
        iteration = 38000
    sample='C'
    last_config = 'config_{0:}_{1:}_{2:}.json'.format(experiment_name, sample, iteration)

    processed = True
    for gpu in range(8):
        if last_config not in processed_configs[gpu]:
            processed = False
    evaluated = True
    if not processed:
        validation_json ='/nrs/saalfeld/heinrichl/synapses/miccai_experiments/{0:}/{1:}.n5/it_{2:}/validation.json'.format(experiment_name, sample, iteration)
        if not os.path.exists(validation_json):
            evaluated = False

    if (not processed) and (not evaluated):
        call(experiment_name)
