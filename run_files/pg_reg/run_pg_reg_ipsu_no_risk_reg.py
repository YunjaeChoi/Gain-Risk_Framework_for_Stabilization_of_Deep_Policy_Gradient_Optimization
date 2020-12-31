from pg_reg import pg_reg
from env_functions import *
from utils.run_utils import ExperimentGrid

if __name__ == '__main__':
    exp_name = 'ipsu-pg_reg-no_risk_reg'

    eg = ExperimentGrid(name=exp_name)
    eg.add('seed', [10*i for i in range(5)])

    eg.add('env_fn', inverted_pendulum_swingup_env_fn, '', False)
    eg.add('epochs', 200)

    eg.add('alpha_ratio', 5.0)
    eg.add('alpha_risk', 5.0)
    eg.add('beta_ratio', 0.1)
    eg.add('beta_risk', 0.0)

    eg.add('save_freq', 100)
    eg.run(pg_reg)
