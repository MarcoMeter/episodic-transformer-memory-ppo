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

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "obs": self.obs,
            "memory_mask": self.memory_mask,
            "memory_index": self.memory_index,
            "memory_indices": self.memory_indices,
        }
        # Convert the memories to a tensor
        self.memories = torch.stack(self.memories, dim=0)

        # Flatten all samples and convert them to a tensor except memories and its memory mask
        self.samples_flat = {}
        for key, value in samples.items():
            self.samples_flat[key] = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.
            
        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // self.n_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "memory_index":
                    # Add the correct episode memories to the concerned mini batch
                    mini_batch["memories"] = self.memories[value[mini_batch_indices]]
                else:
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]