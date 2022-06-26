from gym import spaces
import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, max_episode_length:int, device:torch.device) -> None:
        """
        Args:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
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
        self.num_mem_layers = config["episodic_memory"]["num_layers"]
        self.mem_layer_size = config["episodic_memory"]["layer_size"]

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps))
        self.values = torch.zeros((self.n_workers, self.worker_steps))
        self.advantages = torch.zeros((self.n_workers, self.worker_steps))
        # Episodic memory index buffer
        self.memories = []
        self.memory_mask = torch.zeros((self.n_workers, self.worker_steps, self.max_episode_length), dtype=torch.long)
        self.memory_index = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.long)
        # Episodic memory buffer tensors
        self.memories_arr = torch.zeros((self.n_workers, self.worker_steps, self.max_episode_length, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)

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
            "memory_index": self.memory_index,
            "memory_mask": self.memory_mask,
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
        Arguments:
            num_mini_batches {int} -- Number of the to be sampled mini batches
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