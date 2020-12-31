# Gain-Risk Control Framework for Stabilization of Deep Policy Gradient Optimization

This repository is the official implementation of Gain-Risk Control Framework for Stabilization of Deep Policy Gradient Optimization. Thesis for the degree of master by Yun Jae Choi, department of electrical engineering, Korea University.

## Requirements

To install core requirements:

1. install tensorflow

```setup
conda install tensorflow=1.14
```

2. install gym

```setup
pip install gym==0.17.2
```

3. install PyBullet

```setup
pip install pybullet==2.7.9
```

Others(result plot, etc.):

1. install jupyter notebook

```setup
conda install jupyter notebook
```

2. install seaborn

```setup
conda install seaborn
```

## Training

To train the model(s) in the paper, run files:

1. PPO in simple continuous action environment
```train
bash run_ppo_data_sca.sh
```

2. PPO in PyBullet environments
```train
bash run_ppo.sh
```

3. Vanilla policy gradient(VPG) with smooth minimum risk regilarization and smooth maximum absolute log-policy ratio regularization in PyBullet environments
```train
bash run_pg_reg.sh
```

4. PPO with smooth minimum risk regilarization and smooth maximum absolute log-policy ratio regularization in PyBullet environments
```train
bash run_ppo_reg.sh
```

5. PPO with smooth minimum risk regilarization and adaptive regularization in PyBullet environments
```train
bash run_ppo_reg_adapt.sh
```

#### Methods

  1. pg_reg: policy gradient with proposed regularization methods
  2. ppo: PPO
  3. ppo_reg: PPO with smooth minimum risk regilarization and smooth maximum absolute log-policy ratio regularization
  4. ppo_reg_adapt: PPO with smooth minimum risk regilarization and adaptive regularization

Experiment code is based on [Spinning Up](https://github.com/openai/spinningup) version of PPO. It is modified to increase stability and performance.

## Evaluation

To run the trained policy, run in the notebook:

```eval
run_policy.ipynb
```

To plot training results, run in the notebook:

```eval
plot_performance.ipynb
```

## Pre-trained Models

Pre-trained models are in trained_models folder with (environment name)-(method), e.g. ant-pg_reg

Note:

- There are only one pre-trained model per environments and methods except walker-ppo where
it has two models (local optima and not).

## License

All source code files are licensed under the MIT License except env folder which contains a modified [PyBullet](https://github.com/bulletphysics/bullet3) code.
