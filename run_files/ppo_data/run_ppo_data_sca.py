from ppo_data import ppo
from env_functions import *
from utils.run_utils import ExperimentGrid

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ss", default=0, type=int)
    parser.add_argument("--se", default=5, type=int)
    args = parser.parse_args()

    exp_name = 'sca-ppo'

    eg = ExperimentGrid(name=exp_name)
    eg.add('seed', [i for i in range(args.ss, args.se)])

    eg.add('env_fn', sca_env_fn, '', False)
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 200)

    eg.add('save_freq', 100)
    eg.run(ppo)
