import gym
import numpy as np
import time

class CartPole:
    def __init__(self, mask_velocity = False):
        self._env = gym.make("CartPole-v0")
        self.max_episode_steps = self._env.spec.max_episode_steps
        # Whether to make CartPole partial observable by masking out the velocity.
        if not mask_velocity:
            self._obs_mask = np.ones(4, dtype=np.float32)
        else:
            self._obs_mask =  np.array([1, 0, 1, 0], dtype=np.float32)

    @property
    def observation_space(self):
        return self._env.observation_space
    
    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        return obs * self._obs_mask

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs * self._obs_mask, reward / 100.0, done, info

    def render(self):
        self._env.render()
        time.sleep(0.033)

    def close(self):
        self._env.close()