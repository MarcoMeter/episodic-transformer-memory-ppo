import numpy as np
from gym.spaces import Box
from gymnasium import spaces

class PyTorchEnv():
    """This wrapper reshapes the visual observation to the needs of PyTorch. (W x H x C -> C x W x H)"""

    def __init__(self, env):
        """Defines the shape of the new visual observation.
        
        Arguments:
            env {Env} -- The to be wrapped environment that needs visual observations.
        """
        self._env = env

        # Modify visual observation space
        if self._env.visual_observation_space is not None:
            old_shape = self._env.visual_observation_space.shape
            self._visual_observation_space = spaces.Box(
                    low = 0,
                    high = 1.0,
                    shape = (old_shape[2], old_shape[1], old_shape[0]),
                    dtype = np.float32)
        else:
            self._visual_observation_space = None

    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return Box(low = 0, high = 1.0, shape = (self._env.visual_observation_space.shape[2], self._env.visual_observation_space.shape[1], self._env.visual_observation_space.shape[0]), dtype = np.float32)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._env.vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._env._seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        return self._env.get_episode_trajectory

    def reset(self, reset_params = None):
        """Reset the environment. The provided reset_params is a dictionary featuring reset parameters of the environment such as the seed."""
        vis_obs = self._env.reset(reset_params = reset_params)
        # Swap axes to start with the images' channels, this is required by PyTorch
        if vis_obs is not None:
            vis_obs = np.swapaxes(vis_obs, 0, 2)
            vis_obs = np.swapaxes(vis_obs, 2, 1)
        return vis_obs

    def step(self, action):
        """Executes one step of the agent.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
            {numpy.ndarray} -- Stacked visual observation
            {numpy.ndarray} -- Stacked vector observation
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information retrieved from the environment
        """
        vis_obs, reward, done, info = self._env.step(action)
        # Swap axes to start with the images' channels, this is required by PyTorch
        if vis_obs is not None:
            vis_obs = np.swapaxes(vis_obs, 0, 2)
            vis_obs = np.swapaxes(vis_obs, 2, 1)
        return vis_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()