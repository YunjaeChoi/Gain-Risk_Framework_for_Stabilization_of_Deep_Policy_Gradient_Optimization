from ppo import ppo
from env_functions import *
from utils.run_utils import ExperimentGrid

if __name__ == '__main__':
    exp_name = 'idp-ppo'

    eg = ExperimentGrid(name=exp_name)
    eg.add('seed', [10*i for i in range(5)])

    #pendulum
    eg.add('env_fn', inverted_double_pendulum_env_fn, '', False)
    eg.add('epochs', 100)

    eg.add('save_freq', 100)
    eg.run(ppo)
