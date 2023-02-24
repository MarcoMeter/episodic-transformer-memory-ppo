import gymnasium as gym
import numpy as np
import memory_gym
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
from gymnasium import spaces

class MemoryGymWrapper():
    """
    This class wraps memory-gym environments.
    https://github.com/MarcoMeter/drl-memory-gym
    Available Environments:
        SearingSpotlights-v0
        MortarMayhem-v0
        MortarMayhem-Grid-v0
        MysteryPath-v0
        MysteryPath-Grid-v0
    """
    def __init__(self, env_name, reset_params = None, realtime_mode = False) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Name of the memory-gym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        render_mode = None if not realtime_mode else "debug_rgb_array"
        self._env = gym.make(env_name, disable_env_checker = True, render_mode = render_mode)

        self._realtime_mode = realtime_mode

        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (self._env.observation_space.shape[2], self._env.observation_space.shape[1], self._env.observation_space.shape[0]),
                dtype = np.float32)

    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        self._env.reset()
        return self._env.max_episode_steps

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
        """
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        vis_obs, _ = self._env.reset(seed=self._seed, options=options)
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)

        return vis_obs

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {list} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        vis_obs, reward, done, truncation, info = self._env.step(action)
        vis_obs = np.swapaxes(vis_obs, 0, 2)
        vis_obs = np.swapaxes(vis_obs, 2, 1)

        return vis_obs, reward, done, info
    
    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()