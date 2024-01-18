import numpy as np
import torch

from gym import spaces

class Buffer():
    """The buffer stores and prepares the training data. It supports transformer-based memory policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, max_episode_length:int, device:torch.device) -> None:
        """
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            max_episode_length {int} -- The maximum number of steps in an episode
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"]
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        self.max_episode_length = max_episode_length
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))
        # Episodic memory index buffer
        # Whole episode memories
        # The length of memories is equal to the number of sampled episodes during training data sampling
        # Each element is of shape (max_episode_length, num_blocks, embed_dim)
        self.memories = []
        # Memory mask used during attention
        self.memory_mask = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        # Index to select the correct episode memory from self.memories
        self.memory_index = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        # Indices to slice the memory window
        self.memory_indices = torch.zeros((self.n_workers, self.worker_steps, self.memory_length), dtype=torch.long)
