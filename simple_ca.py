import numpy as np
import gym
from gym import spaces

class SimpleContinuousAction(gym.Env):

    def __init__(self):
        super(SimpleContinuousAction, self).__init__()
        self._num_states = 3

        self.action_space = spaces.Box(low=-1., high=1.0, shape=(1,))
        self.observation_space = spaces.Box(low=0., high=1.0, shape=(self._num_states,))
        self._state = 0
        self._elapsed_steps = 0
        self._max_episode_steps = 10

    def step(self, a):
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        else:
            done = False

        if self._state == 0:
            if (a > -0.4) and (a < -0.1):
                self._state = 1
                reward = 0.2
            else:
                reward = 0.1
        else:
            if (a > 0.1) and (a < 0.4):
                self._state = 0
                reward = 1.0
            else:
                self._state = 0
                reward = -1.0


        obs = np.zeros(self._num_states, dtype=self.observation_space.dtype)
        obs[self._state] = 1.
        obs[-1] = self._elapsed_steps / self._max_episode_steps
        return obs, reward, done, {}

    def get_eval_obs(self):
        eval_obs1 = np.zeros(self._num_states, dtype=self.observation_space.dtype)
        eval_obs1[0] = 1.

        eval_obs2 = np.zeros(self._num_states, dtype=self.observation_space.dtype)
        eval_obs2[ 1] = 1.
        eval_obs2[-1] = 1 / self._max_episode_steps
        return eval_obs1, eval_obs2

    def get_eval_acts(self, num_a):
        eval_acts = np.linspace(-1., 1., num=num_a).reshape(-1,1)
        return eval_acts

    def reset(self):
        self._state = 0
        self._elapsed_steps = 0
        obs = np.zeros(self._num_states, dtype=self.observation_space.dtype)
        obs[self._state] = 1.
        return obs
