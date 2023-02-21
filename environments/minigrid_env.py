import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randint
from minigrid.wrappers import *

class Minigrid():
    """This class wraps Gym Minigrid environments.
    https://github.com/maximecb/gym-minigrid
    Custom Environments:
        MiniGrid-MortarAB-v0
        MiniGrid-MortarB-v0
    Available Environments:
        Empty
            - MiniGrid-Empty-5x5-v0
            - MiniGrid-Empty-Random-5x5-v0
            - MiniGrid-Empty-6x6-v0
            - MiniGrid-Empty-Random-6x6-v0
            - MiniGrid-Empty-8x8-v0
            - MiniGrid-Empty-16x16-v0
        Four rooms
            - MiniGrid-FourRooms-v0
        Door & key
            - MiniGrid-DoorKey-5x5-v0
            - MiniGrid-DoorKey-6x6-v0
            - MiniGrid-DoorKey-8x8-v0
            - MiniGrid-DoorKey-16x16-v0
        Multi-room
            - MiniGrid-MultiRoom-N2-S4-v0
            - MiniGrid-MultiRoom-N4-S5-v0
            - MiniGrid-MultiRoom-N6-v0
        Fetch
            - MiniGrid-Fetch-5x5-N2-v0
            - MiniGrid-Fetch-6x6-N2-v0
            - MiniGrid-Fetch-8x8-N3-v0
        Go-to-door
            - MiniGrid-GoToDoor-5x5-v0
            - MiniGrid-GoToDoor-6x6-v0
            - MiniGrid-GoToDoor-8x8-v0
        Put near
            - MiniGrid-PutNear-6x6-N2-v0
            - MiniGrid-PutNear-8x8-N2-v0
        Red and blue doors
            - MiniGrid-RedBlueDoors-6x6-v0
            - MiniGrid-RedBlueDoors-8x8-v0
        Memory
            - MiniGrid-MemoryS17Random-v0
            - MiniGrid-MemoryS13Random-v0
            - MiniGrid-MemoryS13-v0
            - MiniGrid-MemoryS11-v0
            - MiniGrid-MemoryS9-v0
            - MiniGrid-MemoryS7-v0
        Locked room
            - MiniGrid-LockedRoom-v0
        Key corridor
            - MiniGrid-KeyCorridorS3R1-v0
            - MiniGrid-KeyCorridorS3R2-v0
            - MiniGrid-KeyCorridorS3R3-v0
            - MiniGrid-KeyCorridorS4R3-v0
            - MiniGrid-KeyCorridorS5R3-v0
            - MiniGrid-KeyCorridorS6R3-v0
        Unlock
            - MiniGrid-Unlock-v0
        Unlock Pickup
            - MiniGrid-UnlockPickup-v0
        Blocked unlock pickup
            - MiniGrid-BlockedUnlockPickup-v0
        Obstructed mazed
            - MiniGrid-ObstructedMaze-1Dl-v0
            - MiniGrid-ObstructedMaze-1Dlh-v0
            - MiniGrid-ObstructedMaze-1Dlhb-v0
            - MiniGrid-ObstructedMaze-2Dl-v0
            - MiniGrid-ObstructedMaze-2Dlh-v0
            - MiniGrid-ObstructedMaze-2Dlhb-v0
            - MiniGrid-ObstructedMaze-1Q-v0
            - MiniGrid-ObstructedMaze-2Q-v0
            - MiniGrid-ObstructedMaze-Full-v0
        Distributional shift
            - MiniGrid-DistShift1-v0
            - MiniGrid-DistShift2-v0
        Lava gap
            - MiniGrid-LavaGapS5-v0
            - MiniGrid-LavaGapS6-v0
            - MiniGrid-LavaGapS7-v0
        Lava crossing
            - MiniGrid-LavaCrossingS9N1-v0
            - MiniGrid-LavaCrossingS9N2-v0
            - MiniGrid-LavaCrossingS9N3-v0
            - MiniGrid-LavaCrossingS11N5-v0
        Simple crossing
            - MiniGrid-SimpleCrossingS9N1-v0
            - MiniGrid-SimpleCrossingS9N2-v0
            - MiniGrid-SimpleCrossingS9N3-v0
            - MiniGrid-SimpleCrossingS11N5-v0
        Dynaimc obstacles
            - MiniGrid-Dynamic-Obstacles-5x5-v0
            - MiniGrid-Dynamic-Obstacles-Random-5x5-v0
            - MiniGrid-Dynamic-Obstacles-6x6-v0
            - MiniGrid-Dynamic-Obstacles-Random-6x6-v0
            - MiniGrid-Dynamic-Obstacles-8x8-v0
            - MiniGrid-Dynamic-Obstacles-16x16-v0
    """

    def __init__(self, env_name, realtime_mode = False):
        """Instantiates the Minigrid environment.
        
        Arguments:
            env_name {string} -- Name of the Minigrid environment
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        """
        # Set default reset parameters if none were provided
        self._default_reset_params = {"view-size": 3, "max-episode-steps": 96}

        self._max_episode_steps = self._default_reset_params["max-episode-steps"]
        render_mode = "human" if realtime_mode else "rgb_array"

        # Instantiate the environment and apply various wrappers
        self._env = gym.make(env_name, render_mode=render_mode, agent_view_size=self._default_reset_params["view-size"], tile_size=28)
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = ImgObsWrapper(self._env)

        self._realtime_mode = realtime_mode

        # Prepare observation space
        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, 84, 84),
                dtype = np.float32)

        # Set action space
        if "Memory" in env_name:
            self._action_space = spaces.Discrete(4)
        else:
            self._action_space = self._env.action_space

    @property
    def observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._max_episode_steps

    def reset(self):
        """Resets the environment.
        
        Returns:
            {numpy.ndarray} -- observation
        """
        
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs, _ = self._env.reset(seed=np.random.randint(0, 999))
        # Retrieve the RGB frame of the agent's vision
        obs = obs.astype(np.float32) / 255.
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)

        return obs

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        obs, reward, done, truncated, info = self._env.step(action[0])
        self._rewards.append(reward)
        
        # Retrieve the RGB frame of the agent's vision
        obs = obs.astype(np.float32) / 255.
        
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        
        # Check time limit
        if len(self._rewards) == self._max_episode_steps:
            done = True

        # Wrap up episode information once completed (i.e. done)
        if done or truncated:
            success = 1.0 if sum(self._rewards) > 0 else 0.0
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "success": success}
        else:
            info = None

        return obs, reward, done or truncated, info
    
    def render(self):
        img = self._env.render()
        time.sleep(0.5)
        return img

    def close(self):
        """Shuts down the environment."""
        self._env.close()