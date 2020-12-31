import numpy as np
import gym

class TimeLimitObservation(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimitObservation, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        assert len(self.observation_space.shape) == 1
        obs_size = self.observation_space.shape[0]
        self.observation_space.shape = (obs_size + 1,)

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        observation_time = float(self._elapsed_steps)
        observation_time /= self._max_episode_steps
        observation = np.append(observation, observation_time)
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        observation = self.env.reset(**kwargs)
        return np.append(observation, 0.0)
