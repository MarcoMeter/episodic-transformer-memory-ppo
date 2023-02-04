import gym
import numpy as np
import time

from gym import spaces
from gym_minigrid.wrappers import *
from environments.mortar_env import *

class Minigrid:
    def __init__(self, name):
        self._env = gym.make(name)
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        if "Memory" in name:
            view_size = 3
            self.tile_size = 28
            hw = view_size * self.tile_size
            self.max_episode_steps = 96
            self._action_space = spaces.Discrete(3)
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)
        elif "Mortar" in name:
            self.tile_size = 9
            hw = 9 * self.tile_size
            self._env = RGBImgObsWrapper(self._env, tile_size=self.tile_size)
            self.max_episode_steps = 72
            self._action_space = spaces.Discrete(4)
        else:
            view_size = 7
            self.tile_size = 8
            hw = view_size * self.tile_size
            self.max_episode_steps = 64
            self._action_space = self._env.action_space
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)

        self._env = ImgObsWrapper(self._env)

        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, hw, hw),
                dtype = np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return self._action_space
    
    def reset(self):
        self._env.seed(np.random.randint(0, 999))
        self.t = 0
        self._rewards = []
        obs = self._env.reset()
        obs = obs.astype(np.float32) / 255.

        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.t += 1
        return obs, reward, done, info

    def render(self):
        img = self._env.render(tile_size = 96)
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()