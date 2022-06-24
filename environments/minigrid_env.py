import gym
import numpy as np
import time

from gym import spaces
from gym_minigrid.wrappers import ViewSizeWrapper

class Minigrid:
    def __init__(self, name):
        self._env = gym.make(name)
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        self._env = ViewSizeWrapper(self._env, 3)
        self.max_episode_steps = self._env.max_steps
        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, 84, 84),
                dtype = np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return spaces.Discrete(3)

    def reset(self):
        self._env.seed(np.random.randint(0, 99))
        self._rewards = []
        obs = self._env.reset()
        obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255.
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        obs = self._env.get_obs_render(obs["image"], tile_size=28).astype(np.float32) / 255.
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs, reward, done, info

    def render(self):
        img = self._env.render(tile_size = 96)
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()