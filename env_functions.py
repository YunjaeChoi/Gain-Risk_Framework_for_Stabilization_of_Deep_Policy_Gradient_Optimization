import gym
from wrappers import TimeLimitObservation
from envs.pybullet_envs.gym_locomotion_envs import *
from envs.pybullet_envs.gym_pendulum_envs import *

from simple_ca import SimpleContinuousAction

def half_cheetah_env_fn():
    env = HalfCheetahBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def walker_env_fn():
    env = Walker2DBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def hopper_env_fn():
    env = HopperBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def ant_env_fn():
    env = AntBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def inverted_pendulum_swingup_env_fn():
    env = InvertedPendulumSwingupBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def inverted_double_pendulum_env_fn():
    env = InvertedDoublePendulumBulletEnv()
    env = TimeLimitObservation(env, max_episode_steps=1000)
    return env

def sca_env_fn():
    env = SimpleContinuousAction()
    return env
